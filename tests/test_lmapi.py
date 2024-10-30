from setlexsem.experiment.lmapi import parse_lm_response


def test_parse_lm_response():
    result = " { 'D', 'H', 'P', 'R', 'S', 'f', 'u'}"
    result_obj = parse_lm_response("<answer>" + result + "</answer>")
    assert result_obj == {"P", "R", "D", "u", "H", "f", "S"}
    result = " {f, u, S, P, H, R, D}"
    result_obj = parse_lm_response("<answer>" + result + "</answer>")
    assert result_obj == {"f", "R", "D", "u", "S", "H", "P"}
    result = " {P}"
    result_obj = parse_lm_response("<answer>" + result + "</answer>")
    assert result_obj == {"P"}
    result = " {hello, this, sure}"
    result_obj = parse_lm_response("<answer>" + result + "</answer>")
    assert result_obj == {"sure", "hello", "this"}
    result = " {'2', '4', '7', '6', '5', '3'}"
    result_obj = parse_lm_response("<answer>" + result + "</answer>")
    assert result_obj == {2, 3, 4, 5, 6, 7}
    result = ' {"1", "4", "2"}'
    result_obj = parse_lm_response("<answer>" + result + "</answer>")
    assert result_obj == {1, 2, 4}
    result = " {1, 4, 2}"
    result_obj = parse_lm_response("<answer>" + result + "</answer>")
    assert result_obj == {1, 2, 4}
    result = " {1,'a','hello',5}"
    result_obj = parse_lm_response("<answer>" + result + "</answer>")
    assert result_obj == {"a", 1, 5, "hello"}


if __name__ == "__main__":
    test_parse_lm_response()
    print("All tests passed for lm parser!")
