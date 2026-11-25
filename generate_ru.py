import argparse
import csv
import re
import subprocess
import sys
from zipfile import BadZipFile, ZipFile
from datetime import datetime
from functools import partial
from unittest.mock import patch
from pathlib import Path
from time import perf_counter

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TRANSLATOR = "translate-ru_en-1_9"
TRANSLATOR_URL = "https://argos-net.com/v1/translate-ru_en-1_9.argosmodel"
TRANSLATOR_FILES = (
    "sentencepiece.model",
    "model/shared_vocabulary.json",
    "model/model.bin",
    "model/config.json",
)
SIZE = 512
GUIDANCE = 7.5
DEFAULT_SEED = 42
EXPERIMENT_PROMPTS = (
    ("озеро", "Деревянный домик у озера и осенние деревья, акварель"),
    ("кот", "Рыжий кот на подоконнике, фотография при мягком дневном свете"),
    ("лампа", "Зелёная настольная лампа на белом фоне, предметная фотография"),
)


def positive_integer(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Число шагов должно быть положительным.")
    return value


def seed_number(text):
    value = int(text)
    if not 0 <= value < 2**32:
        raise argparse.ArgumentTypeError("Seed должен быть от 0 до 4294967295.")
    return value


def translator_folder(cache):
    folder = cache / TRANSLATOR
    if all((folder / name).is_file() for name in TRANSLATOR_FILES):
        return folder

    package = cache / f"{TRANSLATOR}.argosmodel"
    if not package.is_file():
        from urllib.request import urlretrieve

        print("Скачиваю пакет перевода.", file=sys.stderr, flush=True)
        cache.mkdir(exist_ok=True)
        partial = package.with_suffix(".partial")
        urlretrieve(TRANSLATOR_URL, partial)
        partial.replace(package)

    with ZipFile(package) as archive:
        for name in TRANSLATOR_FILES:
            archive.extract(f"{TRANSLATOR}/{name}", cache)
    return folder


def translate_prompt(prompt, cache):
    if not re.search(r"[а-яё]", prompt, re.IGNORECASE):
        return prompt

    extra_packages = cache / "translation_packages"
    if extra_packages.is_dir():
        sys.path.insert(0, str(extra_packages))
    from ctranslate2 import Translator
    from sentencepiece import SentencePieceProcessor

    folder = translator_folder(cache)
    tokenizer = SentencePieceProcessor(model_file=str(folder / "sentencepiece.model"))
    tokens = tokenizer.encode(prompt, out_type=str)
    if len(tokens) > 150:
        raise ValueError("Описание слишком длинное для перевода. Сократите его.")

    translator = Translator(str(folder / "model"), device="cpu", intra_threads=4)
    result = translator.translate_batch([tokens], beam_size=4, max_input_length=150)
    translated = tokenizer.decode(result[0].hypotheses[0]).strip()
    if not translated:
        raise ValueError("Перевод пуст. Уточните описание.")
    return translated


def generate(pipe, torch, prompt, steps, seed, path):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = perf_counter()
    with torch.inference_mode():
        result = pipe(prompt, width=SIZE, height=SIZE, guidance_scale=GUIDANCE,
                      num_inference_steps=steps, generator=generator)
    seconds = perf_counter() - started
    flagged = bool(result.nsfw_content_detected and result.nsfw_content_detected[0])
    if not flagged:
        result.images[0].save(path)
    return seconds, flagged


def main():
    parser = argparse.ArgumentParser(description="Генерация изображений Stable Diffusion 1.5 на CPU")
    commands = parser.add_subparsers(dest="mode", required=True)
    single = commands.add_parser("generate", help="Сгенерировать одно изображение")
    single.add_argument("prompt", help="Описание изображения на русском")
    single.add_argument("--steps", type=positive_integer, default=20)
    single.add_argument("--seed", type=seed_number, default=DEFAULT_SEED)
    experiment = commands.add_parser("experiment", help="Сравнить число шагов по трём описаниям")
    experiment.add_argument("--steps", type=positive_integer, nargs="+", default=[10, 20, 30])
    experiment.add_argument("--seed", type=seed_number, default=DEFAULT_SEED)
    if len(sys.argv) == 1:
        prompt = input("Введите описание картинки по-русски и нажмите Enter: ").strip()
        if not prompt:
            print("Описание пустое. Генерация отменена.")
            return
        args = parser.parse_args(["generate", "--", prompt])
    else:
        args = parser.parse_args()
    project = Path(__file__).resolve().parent
    prompts = EXPERIMENT_PROMPTS
    if args.mode == "generate":
        if not re.search(r"[а-яё]", args.prompt, re.IGNORECASE):
            parser.error("Введите описание изображения на русском.")
        prompts = (("изображение", args.prompt),)

    runs = []
    for name, original in prompts:
        # Отдельный процесс освобождает память переводчика до загрузки Stable Diffusion.
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--перевод", original],
            capture_output=True, encoding="utf-8", errors="replace",
        )
        if result.returncode or not result.stdout.strip():
            parser.exit(1, f"Не удалось перевести описание.\n{result.stderr.strip()}\n")
        steps_to_run = (args.steps,) if args.mode == "generate" else dict.fromkeys(args.steps)
        for steps in steps_to_run:
            runs.append((name, original, result.stdout.strip(), steps))

    try:
        import torch
        from diffusers import StableDiffusionPipeline
        from safetensors.torch import load_file
    except ImportError as error:
        parser.exit(1, f"Не удалось импортировать библиотеки. Проверьте установку зависимостей из requirements_ru.txt.\n{error}\n")

    local_model = project / "model_cache" / "modelscope-sd15"
    source = local_model if (local_model / "model_index.json").is_file() else MODEL
    print(f"Загружаю Stable Diffusion 1.5 на CPU: {source}", flush=True)
    try:
        # Читаем веса напрямую с диска, обходя аварийное завершение mmap в Windows.
        with patch("safetensors.torch.load_file", partial(load_file, backend="pread")):
            pipeline = StableDiffusionPipeline.from_pretrained(
                source, torch_dtype=torch.float32, use_safetensors=True,
                cache_dir=project / "model_cache", local_files_only=isinstance(source, Path),
            ).to("cpu")
    except OSError as error:
        parser.exit(1, f"Не удалось загрузить модель. Проверьте доступ к весам.\n{error}\n")
    model_source = "AI-ModelScope/stable-diffusion-v1-5" if isinstance(source, Path) else MODEL

    output = project / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    if args.mode == "experiment":
        print("Пробный запуск перед замерами.", flush=True)
        generate(pipeline, torch, runs[0][2], 1, args.seed, output / "прогрев.png")

    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["file", "steps", "seed", "width", "height", "guidance",
                         "seconds", "status", "model", "device", "prompt_original"])
        for name, original_prompt, prompt, steps in runs:
            path = output / f"{name}_{steps}_steps.png"
            print(f"Генерация: {name}, шагов: {steps}, seed: {args.seed}", flush=True)
            seconds, flagged = generate(pipeline, torch, prompt, steps, args.seed, path)
            writer.writerow(["" if flagged else path.name, steps, args.seed, SIZE, SIZE,
                             GUIDANCE, f"{seconds:.2f}", "filtered" if flagged else "ok",
                             model_source, "cpu", original_prompt])
            file.flush()
            print(f"{seconds:.2f} с. " + ("Изображение отфильтровано." if flagged else path.name))
    print(f"Результаты: {output}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--перевод":
        sys.stdout.reconfigure(encoding="utf-8")
        try:
            print(translate_prompt(sys.argv[2], Path(__file__).resolve().parent / "model_cache"))
        except (ImportError, OSError, RuntimeError, ValueError, BadZipFile) as error:
            sys.exit(str(error))
    else:
        main()
