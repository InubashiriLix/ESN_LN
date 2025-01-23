#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :run_predict.py
# @Time        :2025/1/9 上午9:41
# @Author      :InubashiriLix
import numpy as np

if __name__ == "__main__":
    # the training data is a sin wave
    data = np.sin(np.linspace(0, 6 * np.pi, 101).reshape(-1, 1))
    X_TRAIN = data[:50]
    Y_TRAIN = data[1:51]

    X_TEST = data[50:100]
    Y_TEST = data[51:101]

    from basic_torchless import EsnModel as EsnModel_1
    model_1 = EsnModel_1(100, 0.8, 1e-7)

    from model_trainer import Trainer
    trainer = Trainer(model_1)

    trainer.load_data(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)
    trainer.train()
    trainer.predict(True, False, X_TEST, Y_TEST)

    run_code = 0
