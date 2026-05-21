from cad_ai_suite.cad.parametric_builders import parse_design_prompt


def test_l_bracket_prompt_extracts_hole_count_and_metric_thread() -> None:
    spec = parse_design_prompt(
        "Generate an aluminum L bracket 80mm by 120mm, 6mm thick, "
        "with four M6 holes and a triangular gusset"
    )

    assert spec.kind == "l_bracket"
    assert spec.material == "aluminum"
    assert spec.length_mm == 80.0
    assert spec.height_mm == 120.0
    assert spec.thickness_mm == 6.0
    assert spec.hole_pattern.count == 4
    assert spec.hole_pattern.diameter_mm == 6.0
    assert spec.gusset is True


def test_plate_prompt_extracts_numeric_hole_count() -> None:
    spec = parse_design_prompt("Create a steel plate 100mm by 50mm with 2 M8 holes")

    assert spec.kind == "mounting_plate"
    assert spec.material == "steel"
    assert spec.hole_pattern.count == 2
    assert spec.hole_pattern.diameter_mm == 8.0
