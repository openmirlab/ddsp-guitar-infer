import pathlib

import ddsp_guitar_infer.cli as cli


def test_cli_invokes_synth(monkeypatch, tmp_path):
    midi_path = tmp_path / "input.mid"
    midi_path.write_text("dummy midi")
    output_path = tmp_path / "out.wav"

    captured = {}

    class DummySynth:
        def render_midi(self, **kwargs):
            captured["render"] = kwargs
            path = pathlib.Path(kwargs["output_path"])
            path.write_bytes(b"RIFF")

    def fake_load_synth(**kwargs):
        captured["load"] = kwargs
        return DummySynth()

    monkeypatch.setattr(cli, "load_synth", fake_load_synth)

    exit_code = cli.main(
        [
            str(midi_path),
            "--output",
            str(output_path),
            "--device",
            "cuda:0",
            "--crop-seconds",
            "2",
            "--skip-ratio",
            "0.25",
            "--render-chunk-seconds",
            "4.0",
            "--no-pitch-correction",
            "--legato",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert captured["load"]["device"] == "cuda:0"
    assert captured["render"]["midi_path"] == midi_path
    assert captured["render"]["pitch_correction"] is False
    assert captured["render"]["legato"] is True
    assert captured["render"]["render_chunk_seconds"] == 4.0
