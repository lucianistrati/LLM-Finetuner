# LLM-Finetuner

## Project Description

**LLM-Finetuner** is a Python-based project designed to fine-tune a language model using the Llama-2 architecture. This project utilizes the Hugging Face Transformers library, along with PEFT (Parameter-Efficient Fine-Tuning) techniques to adapt the model for specific instruction-based tasks. The fine-tuned model can be used for various natural language processing applications, such as chatbots, question answering, and more.

## Key Features

- Fine-tuning of the Llama-2 7B model with a custom dataset.
- Utilizes PEFT techniques with LoRA (Low-Rank Adaptation) for efficient training.
- Supports 4-bit quantization for reduced memory usage and faster inference.
- Integrates TensorBoard for monitoring training progress and performance.
- Easy-to-use pipeline for generating text after fine-tuning.

## Requirements

- Python 3.7 or higher
- PyTorch
- Hugging Face Transformers
- Datasets
- PEFT library
- TensorBoard

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/LLM-Finetuner.git
   cd LLM-Finetuner
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**:
   Install the necessary libraries using pip:
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
   pip install transformers datasets peft tensorboard
   ```

## Usage

### Running the Fine-Tuning Script

To start fine-tuning the Llama-2 model using the provided dataset, run the following command:

```bash
python finetune.py
```

### Key Components

- **Model and Dataset Configuration**: The script initializes the base model (`NousResearch/Llama-2-7b-chat-hf`) and loads a custom dataset (`mlabonne/guanaco-llama2-1k`) for fine-tuning.
- **Quantization Configuration**: Utilizes Bits and Bytes to enable 4-bit quantization for memory-efficient model training.
- **PEFT Configuration**: Implements Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.
- **Training Arguments**: Configures various training parameters such as learning rate, batch size, and logging steps.
- **Trainer**: The `SFTTrainer` class is used for training the model with the specified configurations.
- **Text Generation Pipeline**: After fine-tuning, the model can be used to generate text responses to given prompts.

### Example Prompts

After fine-tuning, the script demonstrates how to use the model for text generation:

```python
prompt = "Who is Leonardo Da Vinci?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

prompt = "What is Datacamp Career track?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

## Monitoring Training

The training process can be monitored using TensorBoard. Start TensorBoard with the following command after running the fine-tuning script:

```bash
tensorboard --logdir results/runs --port 4000
```

Open your web browser and navigate to `http://localhost:4000` to view the training metrics.

## Contributing

Contributions are welcome! If you'd like to improve this project, please fork the repository and create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for providing a powerful framework for NLP.
- [PEFT](https://github.com/huggingface/peft) for enabling parameter-efficient fine-tuning.
- [TensorBoard](https://www.tensorflow.org/tensorboard) for visualization of training metrics.
