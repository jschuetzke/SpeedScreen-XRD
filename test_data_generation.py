from generate_training_data import main


def test_simulation():
    main("test", "./test.cif", "./config.yml", n_train=10, n_val=6)
    return
