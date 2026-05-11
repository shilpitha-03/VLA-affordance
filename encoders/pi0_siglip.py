"""Load SigLIP vision encoder from pi0 base checkpoint."""
import torch


def load_pi0_siglip(model_name="lerobot/pi0_base", device="cuda"):
    """Load SigLIP encoder weights from the pi0 base model.

    pi0 is a 3.3B-parameter VLA built on PaliGemma. During flow-matching
    fine-tuning on ~10k hours of manipulation data, the vision tower is
    unfrozen, so its weights diverge from PaliGemma's.

    Returns the *inner* SiglipVisionTransformer (.vision_model), consistent
    with the other SigLIP loaders.
    """
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    policy = PI0Policy.from_pretrained(model_name)

    # Drill in two attribute hops:
    #   .vision_tower            -> SiglipVisionModel (wrapper)
    #   .vision_tower.vision_model -> SiglipVisionTransformer (inner — has .embeddings, .encoder)
    vision_model = (
        policy.model.paligemma_with_expert.paligemma.vision_tower.vision_model
    )
    vision_model = vision_model.to(device).eval()

    # Free the action expert + LLM (~3 GB GPU memory we don't need)
    del policy
    torch.cuda.empty_cache()

    from transformers import SiglipImageProcessor
    processor = SiglipImageProcessor.from_pretrained(
        "google/siglip-so400m-patch14-384"
    )
    processor.size = {"height": 224, "width": 224}
    return vision_model, processor
