"""Tests for trainer wrapper constructors and basic attributes (no GPU needed)."""

from soup_cli.config.schema import SoupConfig


def _make_config(**overrides):
    """Create a minimal SoupConfig for testing."""
    base = {
        "base": "test-model",
        "data": {"train": "./data.jsonl", "format": "alpaca"},
    }
    base.update(overrides)
    return SoupConfig(**base)


class TestSFTTrainerInit:
    """Test SFTTrainerWrapper constructor."""

    def test_default_attributes(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config()
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        assert wrapper.config == cfg
        assert wrapper.device == "cpu"
        assert wrapper.report_to == "none"
        assert wrapper.deepspeed_config is None
        assert wrapper.model is None
        assert wrapper.tokenizer is None
        assert wrapper.trainer is None

    def test_custom_report_to(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config()
        wrapper = SFTTrainerWrapper(cfg, device="cuda", report_to="wandb")
        assert wrapper.report_to == "wandb"

    def test_deepspeed_config(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config()
        wrapper = SFTTrainerWrapper(
            cfg, device="cuda", deepspeed_config="/path/to/ds.json"
        )
        assert wrapper.deepspeed_config == "/path/to/ds.json"


class TestDPOTrainerInit:
    """Test DPOTrainerWrapper constructor."""

    def test_default_attributes(self):
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        cfg = _make_config(task="dpo")
        wrapper = DPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config == cfg
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.ref_model is None
        assert wrapper.tokenizer is None
        assert wrapper.trainer is None

    def test_report_to_wandb(self):
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        cfg = _make_config(task="dpo")
        wrapper = DPOTrainerWrapper(cfg, device="cpu", report_to="wandb")
        assert wrapper.report_to == "wandb"


class TestGRPOTrainerInit:
    """Test GRPOTrainerWrapper constructor."""

    def test_default_attributes(self):
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = _make_config(task="grpo")
        wrapper = GRPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config == cfg
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.tokenizer is None


class TestRewardModelTrainerInit:
    """Test RewardModelTrainerWrapper constructor."""

    def test_default_attributes(self):
        from soup_cli.trainer.reward_model import RewardModelTrainerWrapper

        cfg = _make_config(task="reward_model")
        wrapper = RewardModelTrainerWrapper(cfg, device="cpu")
        assert wrapper.config == cfg
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.tokenizer is None
        assert wrapper.trainer is None


class TestPPOTrainerInit:
    """Test PPOTrainerWrapper constructor."""

    def test_default_attributes(self):
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = _make_config(task="ppo")
        wrapper = PPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config == cfg
        assert wrapper.device == "cpu"


class TestTrainTaskRouting:
    """Test that train command routes to correct trainer based on task."""

    def test_sft_is_default_task(self):
        cfg = _make_config()
        assert cfg.task == "sft"

    def test_dpo_task(self):
        cfg = _make_config(task="dpo")
        assert cfg.task == "dpo"

    def test_grpo_task(self):
        cfg = _make_config(task="grpo")
        assert cfg.task == "grpo"

    def test_ppo_task(self):
        cfg = _make_config(task="ppo")
        assert cfg.task == "ppo"

    def test_reward_model_task(self):
        cfg = _make_config(task="reward_model")
        assert cfg.task == "reward_model"

    def test_backend_default_is_transformers(self):
        cfg = _make_config()
        assert cfg.backend == "transformers"

    def test_backend_unsloth(self):
        cfg = _make_config(backend="unsloth")
        assert cfg.backend == "unsloth"

    def test_modality_default_is_text(self):
        cfg = _make_config()
        assert cfg.modality == "text"

    def test_modality_vision(self):
        cfg = _make_config(modality="vision")
        assert cfg.modality == "vision"


class TestEnableHfTransferProgress:
    """Test _enable_hf_transfer_progress utility."""

    def test_enables_progress_bars(self):
        from soup_cli.trainer.sft import _enable_hf_transfer_progress

        # Should not raise
        _enable_hf_transfer_progress()
