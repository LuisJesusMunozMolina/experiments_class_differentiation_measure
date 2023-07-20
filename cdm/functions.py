# This file contains the code associated to the functions that compute the Class Differentiation Measure
import numpy as np


def distance_to_point(point, array):
    """
    This function computes the distance of one single point (with respect the Frobenious norm) to a set of
    (dimensioanlly coherent) array of points.
    Parameters
    ----------
    point: numpy array. Query point
    array: numpy array. Array of target points. The distance will be computed with respect each point contained in this
    array.

    Returns
    -------
    array of distances.
    """

    dist = []
    for i in range(array.shape[0]):
        d = np.linalg.norm(point - array[i])
        dist.append(d)
    return np.array(dist)


def centroid_distance(sample):
    """
    This function computes the distance from the mass center of a given set to each other point

    Parameter
    ---------
    sample: sample of elements on which the method will be applied.

    Return
    ------
    d_v: distance vector
    """
    m = np.mean(sample, axis=0)
    d_v = distance_to_point(m, sample)
    return d_v


def density_funct_estimation(sample, density_factor=100):
    """
    This function estimates a density function for the distances with respect the centre of mass of a provided sample.

    Parameters
    ----------
    sample: numpy array. Target sample.
    density_factor: int. Steps or divisions in which the density function is computed. Default: 100

    Returns
    -------
    q: numpy array. x_values of the computed density function. These values are computed as the q-quantiles of the
    distances of the samples with respect the centre of mass. q is determined by the density_factor.
    h: numpy array. y_values of the computed density function. These values are computed so that (q_i - q_(i-1))*h_i =
    1/density_factor.
    """
    q = []
    for i in range(1, density_factor):
        q.append(np.quantile(sample, i / density_factor))
    q = np.array(q)
    q = np.concatenate((np.array([np.min(sample)]), q, np.array([np.max(sample)])))
    dq = np.diff(q)
    h = (1 / density_factor) / dq

    return q[1:], h


def mutual_density(target_random_sample, non_target_random_sample, non_target_quantiles, quantile_index = -1):
    """
    This function computes the empirical probability measure of the intersection of two
    categories within a dataset.

    Params
    --------
    target_random_sample: np.array. Samples that belongs to the target category. The
                          probability measure will be computed with respect the empirical
                          probability distribution associated to this category.
    non_target_random_sample: np.array. Samples that belong to some other category. If
                          these two categories are not disjoint then the intersection will
                          be a measurable set and its measure can be computed with respect
                          the different probability measures.
    non_target_quantiles: np.array. Array that contains the quantiles of distances with respect
                          the non_target_random_sample.
    quantile_index: int.  Index of the quantile to be taken as radius from the non_target_quantiles array.

    Return
    -------
    m: float. Measure of the intersection of the two categories with respect the empirical prob
              bability measure defined for the target_random_sample-
    """

    r = non_target_quantiles[quantile_index]
    c2 = np.mean(non_target_random_sample, axis=0)
    distances_to_c2 = distance_to_point(point=c2, array=target_random_sample)
    categories_intersection = distances_to_c2[np.where(distances_to_c2 <= r)]
    m = categories_intersection.shape[0] / target_random_sample.shape[0]
    return m


def mutual_density_divergence(target_category, category_list):
    """
    This function computes a coefficient associated to the measure of
    the intersections of a measurable set with respect someother measurable sets
    for a set of probability measures. This coefficient intends to provide insights
    of the average 'size'of these intersections and, in that way, how distinguishable
    is a set from others.

    Parameters
    -----------
    target_category: array. Dataset associated to the target category for which the coe-
                     fficient will be computed.
    category_list: list of arrays. This list must contain all datasets associated to the
              different categories  in a classification problem.
    mode: int. Used mean method. Default:0 Arithmetic mean. 1, geometric mean.

    Return
    -------
    phi: float. Coefficient associated to the target category.

    """

    relative_measures = []
    for category in category_list:
        d = centroid_distance(category)
        q, h = density_funct_estimation(d)
        m = mutual_density(target_category, category, q, quantile_index=40)
        relative_measures.append(m)
    relative_measures = np.array(relative_measures)
    phi = np.mean(np.log(1 / (1 - relative_measures)))
    return phi, relative_measures
