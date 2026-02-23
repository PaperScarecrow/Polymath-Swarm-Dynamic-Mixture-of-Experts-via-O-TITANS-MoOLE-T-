# Polymath-Swarm-Dynamic-Mixture-of-Experts-via-O-TITANS-MoOLE-T-

The Paradigm Shift The current open-source meta relies on monolithic, massive parameter models (70B+) to achieve multi-domain competency. This approach is computationally expensive, hardware-restrictive, and prone to catastrophic forgetting during fine-tuning.

The Polymath Swarm introduces the MoOLE-T architecture (Mixture-of-Orthogonal-LoRA-Experts with TITANS routing). Instead of one massive brain, we use a lightweight cognitive router to dynamically hot-swap hyper-specialized "Engrams" onto a mid-sized synthesis core in real-time.

The Architecture

The Brainstem (Cognitive Router): Powered by gemma-3-4b-it. It intercepts the user prompt, utilizes a <think> block to decompose the task, and fires a deterministic routing token (e.g., [ROUTE: code_python]).

The Orchestrator: A localized Python controller that catches the routing token, retrieves the required skill from an engrams.json dictionary, and hot-swaps the physical weights into VRAM in milliseconds.

The Frontal Lobe (Synthesis Core): Powered by gemma-3-12b-it. It acts as the execution engine. It idles in a sterile, baseline state until the Orchestrator mounts a specialized engram to its attention matrices.

The Engrams (O-TITANS Tools): These are not standard LoRAs. They are forged using the Orthogonal-TITANS matrix penalty we published previously [Link to O-TITANS post]. By strictly isolating the fine-tune to the q_proj and v_proj layers and mathematically punishing dimensional overlap, we can inject extreme domain expertise (like advanced Python asyncio networking) without degrading the model's foundational conversational alignment.

The Vision: An "App Store" for Cognition Included in this repository is our first production engram: otitans_code_python.pt.

However, the true goal of the MoOLE-T framework is the creation of a community-driven repository of hot-swappable skills. Users shouldn't have to download a new 20GB model just because they want their AI to analyze medical documents or write cyberpunk fiction. You should be able to download a 25MB .pt file, drop it into your /adapters/ folder, update your engrams.json, and instantly grant your Swarm a new capability.

The Roadmap: "Featherweight" Edge Deployment While this V1 release utilizes a 4B/12B dynamic, we are actively developing the "Nano" variant. By deep-frying the gemma-3-270m-it into a pure stimulus-response Reflex Arc, we will bring this dynamic Mixture-of-Experts architecture to CPU-only and edge devices.

Links & Assets

O-TITANS Gemma 3 Adapters (Proof of Concept): (https://huggingface.co/paperscarecrow/O-TITANS-Gemma3)
Training Scripts & Surgery Methodology: (https://github.com/PaperScarecrow/O-TITANS)

Credits & Resources

A massive credit to the foundational work that made this possible:

ffurfaro for the TPTT "titanesque" methodologies that inspired the titanized-lora structural approach.
mlabonne for the BF16 Gemma-3-abliteration models. The zeroed vectors from his minosv1 process are what make the underlying synthesis actually work without semantic contamination.
Google for the TITANS research
