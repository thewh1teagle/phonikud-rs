/*
Run with:
    wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx -O phonikud.onnx
    wget https://huggingface.co/dicta-il/dictabert-large-char-menaked/raw/main/tokenizer.json -O tokenizer.json
    cargo run --example usage
*/

use phonikud_rs::Phonikud;

fn main() -> anyhow::Result<()> {
    let model_path = "phonikud.onnx";
    let tokenizer_path = "tokenizer.json";

    let mut phonikud = Phonikud::new(model_path, tokenizer_path)?;

    let text = "הכוח לשנות מתחיל ברגע שבו אתה מאמין שזה אפשרי!";
    let vocalized = phonikud.add_diacritics(text)?;

    println!("Input: {}", text);
    println!("Output: {}", vocalized);

    Ok(())
}
