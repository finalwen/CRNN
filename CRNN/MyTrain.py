from crnn import CRNN

CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'.!?,\"&"

crnn = CRNN(
    32,
    './save/',
    'D:/tmp/lstm_ctc_data',
    256,
    0.75,
    './save/',
    char_set_string=CHAR_VECTOR,
    use_trdg=False,
    language='en',
)
crnn.train(1000)
