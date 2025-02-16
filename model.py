import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def setup_logging():
    """Configure logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def load_summarizer(model_name="microsoft/phi-2", save_directory="./phi2_model"):
    """
    Load the summarization model. If not available locally, it will download from Hugging Face.
    """
    try:
        # Try loading from local directory if model is already downloaded
        model = AutoModelForCausalLM.from_pretrained(save_directory)
        tokenizer = AutoTokenizer.from_pretrained(save_directory)
        logging.info(f"Loaded model from local directory: {save_directory}")
    except Exception:
        logging.warning(f"Local model not found. Downloading {model_name} from Hugging Face...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Save locally for future use
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        logging.info(f"Model downloaded and saved to: {save_directory}")

    # Load text generation pipeline
    summarizer = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    return summarizer

def summarize_text(summarizer, text, max_length=150, min_length=50):
    """
    Generate a summary of the given text.
    """
    prompt = f"""Instruction: Please provide a concise summary of the following text.

    Text: {text}

    Summary:"""

    try:
        result = summarizer(
            prompt,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )[0]['generated_text']

        return result.split("Summary:")[-1].strip()
    except Exception as e:
        logging.error(f"Failed to generate summary: {e}")
        raise

def main():
    setup_logging()
    logging.info("Loading summarization model...")

    try:
        summarizer = load_summarizer()
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        return

    # Example text to summarize
    text = """Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to the natural intelligence displayed by humans or animals. 
    It is a rapidly growing field with applications in various domains, including 
    healthcare, finance, robotics, and entertainment."""

    logging.info("Generating summary...")
    summary = summarize_text(summarizer, text)

    print("\nOriginal Text:\n", text)
    print("\nGenerated Summary:\n", summary)

if __name__ == "__main__":
    main()
