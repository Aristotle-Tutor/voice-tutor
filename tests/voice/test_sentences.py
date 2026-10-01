from packages.voice.sentences import SentenceBuffer


def test_sentences_come_out_whole_and_keep_every_character() -> None:
    buffer = SentenceBuffer()
    pieces = ["Hi the", "re! How are", " you? I'm fi", "ne.\nOk", "ay then"]

    sentences = [sentence for piece in pieces for sentence in buffer.add(piece)]
    rest = buffer.flush()

    assert sentences == ["Hi there! ", "How are you? ", "I'm fine.\n"]
    assert rest == "Okay then"


def test_decimals_do_not_end_a_sentence() -> None:
    buffer = SentenceBuffer()
    assert buffer.add("Pi is about 3.14 and ") == []
