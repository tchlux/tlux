import subprocess
import shutil
import platform
from transformers import AutoTokenizer
from pathlib import Path

def main():
    # Base directory of this script
    script_dir = Path(__file__).resolve().parent

    # Model ID and temporary folders
    model_id = "facebook/contriever"
    onnx_temp  = script_dir / "onnx-temp"
    quant_temp = script_dir / "onnx-int8-temp"

    # Create temp dirs
    onnx_temp.mkdir(parents=True, exist_ok=True)
    quant_temp.mkdir(parents=True, exist_ok=True)

    # Save tokenizer into the ONNX temp folder
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(onnx_temp)

    # Export ONNX model to the ONNX temp folder
    subprocess.run([
        "optimum-cli", "export", "onnx",
        "--model", model_id,
        "--task", "feature-extraction",
        "--opset", "17",
        str(onnx_temp)
    ], check=True)

    # Detect hardware and choose quantization flag
    arch = platform.machine().lower()
    if arch in ("arm64", "aarch64"):
        quant_flag = "--arm64"
    else:
        quant_flag = "--avx2"

    # Quantize ONNX model into the quant temp folder
    subprocess.run([
        "optimum-cli", "onnxruntime", "quantize",
        "--onnx_model", str(onnx_temp),
        "-o", str(quant_temp),
        quant_flag
    ], check=True)

    # Move only the single tokenizer.json into the project root
    src_file = onnx_temp / "tokenizer.json"
    dst_file = script_dir / "tokenizer.json"
    if dst_file.exists():
        dst_file.unlink()
    shutil.move(src_file, dst_file)

    # Move the quantized ONNX file to script dir
    onnx_files = list(quant_temp.glob("*.onnx"))
    if len(onnx_files) != 1:
        raise RuntimeError(f"Expected one .onnx file in {quant_temp}, found {onnx_files}")
    quant_model = onnx_files[0]
    shutil.move(str(quant_model), str(script_dir / quant_model.name))

    # Clean up temporary folders
    shutil.rmtree(onnx_temp)
    shutil.rmtree(quant_temp)

    print(f" Tokenizer moved to: {dst_file}")
    print(f" Quantized model moved to: {quant_model.name}")

if __name__ == "__main__":
    main()
