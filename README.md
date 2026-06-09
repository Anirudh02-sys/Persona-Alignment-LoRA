# Friends-Finetuned 

Do LLM's have their own personality ? 

I try dapping a try at that by fintuning models to characters in friends and making them take personality assessments. 

> **Core Focus:** LLM Fine-Tuning, Character Persona Alignment, Parameter-Efficient Adaptation
> **Base Architectures:** Compatible with Llama-3 / Mistral-7B / Llama-2-7B
> **Toolchain:** PyTorch, Hugging Face Transformers, PEFT (LoRA), TRL (SFTTrainer), Unsloth

An end-to-end conversational engineering pipeline designed to fine-tune open-source Large Language Models on conversational scripts. This framework handles the entire lifecycle: from raw multi-turn dialogue dataset parsing and prompt-template structuring to Low-Rank Adaptation (LoRA) fine-tuning and inference evaluation.

---

## 🏗️ Pipeline Architecture

```mermaid
graph TD
    A[Raw Script Corpus] --> B(Multi-Turn Dialogue Parser)
    B --> C[Instruction-Prompt Formatter]
    C --> D{Quantized Base LLM}
    D --> E[PEFT / LoRA Adapter Training]
    E --> F[WandB Loss & Token Convergence Tracking]
    F --> G[Character Persona Inference Node]
