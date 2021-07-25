#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 21:25:37 2021
@author: rga
@contact: roy_gonzalez@fq.uh.cu, roy.gonzalez-aleman@u-psud.fr
"""
import pytest
import pickle


def unpickle_from_file(file_name):
    ''' Unserialize a **pickle** file.

    Args:
        file_name (str): file to unserialize.
    Returns:
        (object): an unserialized object.
    '''
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data


golden = 'aligned_tau_mdscan_fd94b7f3.pick'
target = '../aligned_tau_mdscan.pick'
Kd_arr, dist_arr, nn_arr, exhausted, selected, final_array = unpickle_from_file(golden)
Td_arr, tist_arr, tr_arr, txhausted, telected, tinal_array = unpickle_from_file(target)


def test_core_distances():
    assert (Kd_arr == Td_arr).sum() == Kd_arr.size


def test_distance_array():
    assert (dist_arr == tist_arr).sum() == tist_arr.size


def test_nearest_neighbor_array():
    assert (nn_arr == tr_arr).sum() == nn_arr.size


def test_exhausted_nodes():
    assert [x[1] for x in exhausted] == [x[1] for x in txhausted]


def test_selected_clusters():
    assert (selected == telected).sum() == selected.size


def test_final_clustering():
    assert (final_array == tinal_array).sum() == final_array.size
