from inference import ChatInference


def test_adds_default_closing_for_goodbye_or_thanks():
    inference = ChatInference()

    response = inference.append_default_closing("Gracias por tu ayuda")
    assert "Estoy aquí para servirte siempre que lo necesites." in response

    response = inference.append_default_closing("Hasta luego")
    assert "Estoy aquí para servirte siempre que lo necesites." in response

    response = inference.append_default_closing("Necesito ayuda")
    assert "Estoy aquí para servirte siempre que lo necesites." not in response
