from inference import ChatInference


def test_adds_default_closing_for_goodbye_or_thanks():
    inference = ChatInference()

    # "gracias" solo cuenta como despedida cuando lo dice el usuario: el generador lo usa
    # todo el tiempo como apertura empática ("gracias por compartir eso"), así que en la
    # respuesta del bot ese marcador no debe disparar el cierre (ver should_add_default_closing).
    response = inference.append_default_closing("Alguna respuesta generada", user_text="Gracias por tu ayuda")
    assert response in inference.default_farewell_messages

    response = inference.append_default_closing("Hasta luego")
    assert response in inference.default_farewell_messages

    response = inference.append_default_closing("Necesito ayuda")
    assert response not in inference.default_farewell_messages
