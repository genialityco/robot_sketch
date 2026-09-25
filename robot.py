#!/usr/bin/python
# -*- coding:utf-8 -*-
# Conexion unica al brazo, compartida por el dibujo y la calibracion.
#
# El puerto serie solo se puede abrir una vez: antes cada dibujo abria su
# propia conexion, pero la calibracion necesita una que dure entre pedidos
# (liberar servos -> mover a mano -> leer posicion). LOCK asegura que nunca
# haya dos operaciones mandando comandos al brazo a la vez; BUSY dice cual lo
# tiene tomado, para dar un error claro.

import threading

import pipeline as pl

LOCK = threading.Lock()
BUSY = {'what': None}
_STATE = {'mc': None}


class RobotBusy(Exception):
    pass


def get():
    if _STATE['mc'] is None:
        _STATE['mc'] = pl.lib.connect()
    return _STATE['mc']


def is_connected():
    return _STATE['mc'] is not None


def disconnect():
    mc, _STATE['mc'] = _STATE['mc'], None
    if mc is not None:
        try:
            mc.close()
        except Exception:
            pass


def acquire(what):
    """Toma el brazo sin esperar; lanza RobotBusy si ya esta en uso."""
    if not LOCK.acquire(blocking=False):
        raise RobotBusy('el brazo esta ocupado (%s). Espera a que termine.' % (BUSY['what'] or 'otra tarea'))
    BUSY['what'] = what


def release():
    BUSY['what'] = None
    LOCK.release()


class use:
    """with robot.use('leer posicion') as mc: ... -- toma el brazo, conecta
    si hace falta y lo suelta al salir. Si falla la comunicacion, descarta la
    conexion para que el proximo intento reconecte de cero."""

    def __init__(self, what):
        self.what = what

    def __enter__(self):
        acquire(self.what)
        try:
            return get()
        except Exception:
            disconnect()
            release()
            raise

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and not issubclass(exc_type, (ValueError, RobotBusy)):
            disconnect()
        release()
        return False
