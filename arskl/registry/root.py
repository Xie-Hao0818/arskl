from .registry import Registry

DATASETS = Registry('dataset')
LEARNER = Registry('learner')
MODEL = Registry('model')
OPTIM = Registry('optim')
SCHEDULER = Registry('scheduler')
TRAINER = Registry('trainer')
TRANSFORM = Registry('transform')
RECOGNIZER = Registry('recognizer')
