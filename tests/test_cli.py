from pathlib import Path

import numpy as np
import pytest

from forest_fire.cli import main


def test_run_saves_fire_sizes_and_reports_a_fit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "fires.npz"
    main(["run", "--size", "32", "--theta", "20", "--events", "3000", "--burn-in", "500",
          "--seed", "1", "--out", str(out)])  # fmt: skip

    data = np.load(out)
    assert data["fire_sizes"].size > 0
    assert data["tree_density"].size == 3000
    assert "tau =" in capsys.readouterr().out


def test_animate_saves_a_gif(tmp_path: Path) -> None:
    out = tmp_path / "fire.gif"
    main(["animate", "--size", "16", "--steps", "5", "--seed", "1", "--save", str(out)])

    assert out.read_bytes()[:6] in (b"GIF87a", b"GIF89a")


def test_rejects_unknown_neighbourhood() -> None:
    with pytest.raises(SystemExit):
        main(["animate", "--neighbourhood", "hexagonal", "--save", "x.gif"])
