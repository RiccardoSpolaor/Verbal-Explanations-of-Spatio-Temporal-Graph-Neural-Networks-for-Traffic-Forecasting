from datetime import datetime, timedelta
import random
from typing import Any, Dict, List, Tuple

import numpy as np

from .templates import *

from .content_extraction import (
    get_cluster_type,
    get_repetition_of_cluster_type_information,
    get_cluster_time_info,
    get_cluster_day_span,
    get_cluster_location_info,
    get_repetition_of_location_information)

def translate_dataset(
    x: np.ndarray,
    x_times: np.ndarray,
    x_clusters: np.ndarray,
    y: np.ndarray,
    y_times: np.ndarray,
    node_pos_dict: Dict[int, str],
    node_info: Dict[str, Tuple[str, float]],
    ) -> np.ndarray:
    """
    Translate the dataset.
    
    Parameters
    ----------
    x : ndarray
        The input data.
    x_times : ndarray
        The input data times.
    x_clusters : ndarray
        The input data clusters.
    y : ndarray
        The target data.
    y_times : ndarray
        The target data times.
    node_pos_dict : { int: str }
        The dictionary containing the position of each node ID.
    node_info : { str: (str, float) }
        The dictionary containing the information of street and kilometrage
        about each node ID.
        
    Returns
    -------
    ndarray
        The translated dataset.
    """
    translations = []
    for x_, x_times_, x_clusters, y_, y_times_ in zip(x, x_times, x_clusters, y, y_times):
        translations.append(get_verbal_explanation(
            x_,
            x_times_,
            x_clusters,
            y_,
            y_times_,
            node_pos_dict,
            node_info))

    return np.array(translations)
    
def get_verbal_explanation(
    x: np.ndarray,
    x_times: np.ndarray,
    x_clusters: np.ndarray,
    y: np.ndarray,
    y_times: np.ndarray,
    node_pos_dict: Dict[int, str],
    node_info: Dict[str, Tuple[str, float]],
    ) -> str:
    """
    Get the verbal explanation of the prediction.

    Parameters
    ----------
    x : ndarray
        The input data.
    x_times : ndarray
        The input data times.
    x_clusters : ndarray
        The input data clusters.
    y : ndarray
        The target data.
    y_times : ndarray
        The target data times.
    node_info : { str: (str, float) }
        The dictionary containing the information of street and kilometrage
        about each node ID.

    Returns
    -------
    str
        The verbal explanation of the prediction.
    """

    # Get the values of the selected target nodes.
    target_node_values = y[y > 0]

    # Get the type of the target nodes cluster (eg.: congestion, free flow).
    target_cluster_type = get_cluster_type(target_node_values)

    # Get the indices of the non-null values of the target nodes.
    y_indices = np.nonzero(y)

    # Translate the temporal information.
    target_temporal_information = get_cluster_time_info(y_times, y_indices[0])
    day_sentence = _get_target_day_sentence(target_temporal_information)
    time_sentence = _get_time_sentence(target_temporal_information)

    # Translate the location information.
    target_street_information = get_cluster_location_info(
        node_info,
        y_indices[1],
        node_pos_dict)
    street_sentence = _get_target_location_sentence(target_street_information)

    # Get the average speed of the target nodes.
    target_average_speed = target_node_values.mean()

    # Get the first paragraph.
    first_paragraph = _fill_first_paragraph_template(
        target_cluster_type,
        time_sentence,
        day_sentence,
        target_average_speed,
        street_sentence)

    # Set the list of the other paragraphs.
    other_paragraphs = []

    # Get the input clusters IDs.
    input_clusters_ids = [c for c in np.unique(x_clusters) if c != -1]

    input_clusters_with_information = []

    for c in input_clusters_ids:
        # Get the values of the nodes of the cluster.
        input_node_values = x[x_clusters == c]

        # Get the type of the cluster.
        input_cluster_type = get_cluster_type(input_node_values)

        # Get the indices of the clusters in the input data.
        x_indices = np.where(x_clusters == c)

        # Get the temporal information.
        input_temporal_information = get_cluster_time_info(
            x_times, x_indices[0])

        # Get the location information.
        input_location_information = get_cluster_location_info(
            node_info, x_indices[1], node_pos_dict)

        # Get the average speed of the target nodes.
        input_average_speed = input_node_values.mean()

        # Set the input information in a dictionary.
        input_information = {
            'type': input_cluster_type,
            'temporal': input_temporal_information,
            'location': input_location_information,
            'average_speed': input_average_speed
        }

        # Add the cluster information to the list.
        input_clusters_with_information.append((c, input_information))

    first_paragraph = _get_first_paragraph_plus_end_sentence(
        first_paragraph,
        [inf['type'] for _, inf in input_clusters_with_information])

    # Sort the clusters by the time they occur and get just the information.
    input_clusters_with_information = sorted(
        input_clusters_with_information,
        key=lambda x: get_cluster_day_span(x[1]['temporal']))

    for i, (_, info) in enumerate(input_clusters_with_information):
        # Get the type of the input cluster.
        input_cluster_type = info['type']
        same_cluster_type_count = get_repetition_of_cluster_type_information(
            input_cluster_type,
            [inf['type'] for _, inf in input_clusters_with_information[:i]])
        formatted_cluster_type = f'contributing {info["type"]}'
        if i > 0 and same_cluster_type_count == 0:
            formatted_cluster_type = f'a {formatted_cluster_type}'
        elif i > 0 and same_cluster_type_count == 1:
            formatted_cluster_type =\
                f'{random.choice(another_connectors)} {formatted_cluster_type}'
        elif i > 0 and same_cluster_type_count > 1:
            formatted_cluster_type =\
                f'yet {random.choice(another_connectors)} {formatted_cluster_type}'

        if i == 0 and len(input_clusters_ids) == 1:
            paragraph_connector =\
                f'The {{c}} {random.choice(second_paragraph_verbs)}'
        elif i == 0:
            paragraph_connector =\
                f'{random.choice(second_paragraph_connectors)} {random.choice(second_paragraph_verbs)}'
        elif i == len(input_clusters_ids) - 1:
            paragraph_connector =\
                f'{random.choice(final_paragraph_connectors)} {random.choice(second_paragraph_verbs)}'
        else:
            paragraph_connector =\
                f'{random.choice(other_paragraphs_connectors)} {random.choice(second_paragraph_verbs)}'

        # Translate the temporal information.
        day_sentence = _get_input_day_sentence(
            target_temporal_information,
            info['temporal'])
        time_sentence = _get_time_sentence(info['temporal'])

        # Translate the location information.
        previous_location_information = [
            inf['location'] for _, inf in input_clusters_with_information[:i]]
        input_location_sentence = _get_target_location_sentence(
            info['location'],
            previous_location_information)

        # Get the average speed of the target nodes.
        input_average_speed = info['average_speed']

        # Get the other paragraph.
        other_paragraph = _fill_other_paragraph(
            input_cluster_type,
            formatted_cluster_type,
            paragraph_connector,
            time_sentence,
            day_sentence,
            input_average_speed,
            input_location_sentence)

        other_paragraphs.append(other_paragraph)

    # Get the explanation.
    explanation = first_paragraph + '\n\n' + '\n\n'.join(other_paragraphs)
    return explanation

def _replace_template_placeholder(
    sentence: str,
    placeholder: str,
    replacement: str,
    ) -> str:
    """
    Replace the `placeholder` in the `sentence` with the `replacement`.

    Parameters
    ----------
    sentence : str
        The sentence containing the placeholder.
    placeholder : str
        The placeholder to replace.
    replacement : str
        The replacement of the placeholder.

    Returns
    -------
    str
        The sentence with the placeholder replaced.
    """
    if placeholder == '{d}' and replacement == '':
        sentence = sentence.replace('{d}, ', '')
        sentence = sentence.replace(' {d}.', '.')

    # Substitute `placeholder` in the sentence with the `replacement`.
    sentence = sentence.replace(placeholder.upper(), replacement.capitalize())
    sentence = sentence.replace(placeholder, replacement)

    return sentence

def _fill_first_paragraph_template(
    predicted_cluster_kind: str,
    time_sentence: str,
    day_sentence: str,
    average_speed: float,
    street_sentences: List[str],
    ) -> str:
    """
    Fill the template of the first paragraph.

    Parameters
    ----------
    predicted_cluster_kind : str
        The type of the predicted cluster.
    time_sentence : str
        The sentence containing the time information.
    day_sentence : str
        The sentence containing the day information.
    average_speed : float
        The average speed of the target nodes in the cluster.
    street_sentences : list of str
        The list of the sentences containing the location information.

    Returns
    -------
    str
        The filled template of the first paragraph.
    """
    sentence = random.choice(first_paragraph_sentences)
    # substitute prediction verb
    sentence = sentence.replace('{prediction}', random.choice(prediction_verbs))
    # Add the cluster type information to the sentence.
    sentence = _replace_template_placeholder(sentence, '{c}', predicted_cluster_kind)
    # Add the name of the street information to the sentence.
    sentence = _replace_template_placeholder(sentence, '{w}', street_sentences[0][0])
    sentence = _replace_template_placeholder(sentence, '{w:}', street_sentences[0][1])
    # Add the day information to the sentence.
    sentence = _replace_template_placeholder(sentence, '{d}', day_sentence)
    # Add the time information to the sentence.
    sentence = _replace_template_placeholder(sentence, '{t}', time_sentence)
    # Add the average speed information to the sentence.
    sentence = sentence.replace('{s}', f'{average_speed:.2f}')
    if len(street_sentences) > 1:
        extra_involved_street_sentence = random.choice(extra_involved_street_sentences)
        other_locations_sentences = _link_other_locations_sentences(street_sentences[1:])
        other_locations_sentence = other_locations_sentences[0]
        if not other_locations_sentence.startswith(','):
            other_locations_sentence = ' ' + other_locations_sentence
        other_locations_sentence_adv = other_locations_sentences[1]
        extra_involved_street_sentence = _replace_template_placeholder(
            extra_involved_street_sentence, '{w}', other_locations_sentence)
        extra_involved_street_sentence = _replace_template_placeholder(
            extra_involved_street_sentence, '{w:}',
            other_locations_sentence_adv)
        extra_involved_street_sentence = _replace_template_placeholder(
            extra_involved_street_sentence, '{c}', predicted_cluster_kind)
        return sentence + ' ' + extra_involved_street_sentence
    return sentence

def _get_first_paragraph_plus_end_sentence(
    first_paragraph: str,
    cluster_types: List[str]
    ) -> str:
    """
    Get the first paragraph end sentence.

    Parameters
    ----------
    first_paragraph : str
        The first paragraph.
    cluster_types : list of str
        The list of the cluster types present in the explanation.

    Returns
    -------
    str
        The first paragraph with the added end sentence.
    """
    first_paragraph_end_sentence = random.choice(first_paragraph_end_sentences)
    first_paragraph += ' ' + first_paragraph_end_sentence

    n_congestions = 0.
    n_free_flows = 0.

    for cluster_type in cluster_types:
        if cluster_type in['congestion', 'severe congestion']:
            n_congestions += 1
        elif cluster_type == 'free flow':
            n_free_flows += 1
    if (n_congestions, n_free_flows) == (0, 0):
        return first_paragraph + 'n unclear reasons.'
    elif (n_congestions, n_free_flows) == (0, 1):
        return first_paragraph + ' free flow.'
    elif (n_congestions, n_free_flows) == (1, 0):
        return first_paragraph + ' congestion.'
    elif (n_congestions, n_free_flows) == (1, 1):
        return first_paragraph + ' congestion and a free flow.'
    elif (n_congestions, n_free_flows) == (0, len(cluster_types)):
        return first_paragraph + ' series of free flows.'
    elif (n_congestions, n_free_flows) == (len(cluster_types), 0):
        return first_paragraph + ' series of congestions.'
    elif (n_congestions, n_free_flows) == (1, len(cluster_types) - 1):
        return first_paragraph + ' series of free flows and a congestion.'
    elif (n_congestions, n_free_flows) == (len(cluster_types) - 1, 1):
        return first_paragraph +' series of congestions and a free flow.'
    else:
        return first_paragraph + ' series of congestions and free flows.'

def _fill_other_paragraph(
    input_cluster_kind: str,
    formatted_cluster_type: str,
    connector: str,
    time_sentence: str,
    day_sentence: str,
    average_speed: float,
    street_sentences: List[str],
    ) -> str:
    """
    Fill the template of the other paragraphs.

    Parameters
    ----------
    input_cluster_kind : str
        The type of the input cluster.
    formatted_cluster_type : str
        The type of the input cluster formatted.
    connector : str
        The connector of the paragraph.
    time_sentence : str
        The sentence containing the time information.
    day_sentence : str
        The sentence containing the day information.
    average_speed : float
        The average speed of the target nodes in the cluster.
    street_sentence : list of str
        The list of the sentences containing the location information.

    Returns
    -------
    str
        The filled template of the other paragraph.
    """
    sentence = connector + random.choice(second_paragraph_sentences)
    # Add the input type information to the sentence.
    sentence = _replace_template_placeholder(sentence, '{c}', formatted_cluster_type)
    # substitute prediction verb
    sentence = sentence.replace('{prediction}', random.choice(prediction_verbs))
    # Add the name of the street information to the sentence.
    sentence = _replace_template_placeholder(sentence, '{w}', street_sentences[0][0])
    sentence = _replace_template_placeholder(sentence, '{w:}', street_sentences[0][1])
    # Add the day information to the sentence.
    sentence = _replace_template_placeholder(sentence, '{d}', day_sentence)
    # Add the time information to the sentence.
    sentence = _replace_template_placeholder(sentence, '{t}', time_sentence)
    # Add the average speed information to the sentence.
    sentence = sentence.replace('{s}', f'{average_speed:.2f}')
    if len(street_sentences) > 1:
        extra_involved_street_sentence = random.choice(extra_involved_street_sentences)
        other_locations_sentences = _link_other_locations_sentences(street_sentences[1:])
        other_locations_sentence = other_locations_sentences[0]
        if not other_locations_sentence.startswith(','):
            other_locations_sentence = ' ' + other_locations_sentence
        other_locations_sentence_adv = other_locations_sentences[1]
        extra_involved_street_sentence = _replace_template_placeholder(
            extra_involved_street_sentence, '{w}', other_locations_sentence)
        extra_involved_street_sentence = _replace_template_placeholder(
            extra_involved_street_sentence, '{w:}',
            other_locations_sentence_adv)
        extra_involved_street_sentence = _replace_template_placeholder(
            extra_involved_street_sentence, '{c}', input_cluster_kind)
        return sentence + ' ' + extra_involved_street_sentence
    return sentence

def _get_target_location_sentence(
    location_information: Dict[str, List[int]],
    previous_location_information: List[Dict[str, List[int]]] = None
    ) -> List[Tuple[str, str]]:
    """
    Get a list of tuple of sentences which for each involved street contains the
    information about the involved kms. The first sentence of the tuple is the
    sentence without the adverb, the second one is the sentence with the adverb
    "on".

    Parameters
    ----------
    location_information : { str: list of int }
        The dictionary containing the location information.
        Keys are the streets and values are the involved kms.
    previous_location_information : list of { str: list of int }, optional
        The list of the previous location information, by default None

    Returns
    -------
    list of (str, str)
        The list of the sentences containing the location information.
        The first sentence of the tuple is the sentence without the adverb,
        the second one is the sentence with the adverb "on".
    """
    previous_locations_information_count = get_repetition_of_location_information(
        location_information,
        previous_location_information)

    # Set the list of the location sentences.
    location_sentences = []
    for street, kms in location_information.items():
        if previous_locations_information_count[street] == 0:
            connector = ''
        elif previous_locations_information_count[street] == 1:
            connector = f', {random.choice(again_connectors)}, '
        else:
            connector = f', {random.choice(yet_again_connectors)}, '

        location_sentence = f'{connector}{street} at '

        if len(kms) == 1:
            location_sentence += f'km {kms[0]}'
        else:
            kms_sentence = ', '.join([f'{km}' for km in kms[:-1]])
            kms_sentence += f' and {kms[-1]}'
            location_sentence += f'kms {kms_sentence}'
        location_sentences.append((
            location_sentence,
            'on ' + location_sentence if connector == '' else 'on' + location_sentence))
    return location_sentences

def _link_other_locations_sentences(
    location_sentences: List[str],
    ) -> Tuple[str, str]:
    """
    Link the sentences containing the location information in a single
    sentence.

    Parameters
    ----------
    location_sentences : list of str
        The list of the sentences containing the location information.

    Returns
    -------
    str
        The linked sentence containing the location information.
    str
        The linked sentence containing the location information with the
        adverb "on". 
    """
    if len(location_sentences) == 1:
        return location_sentences[0]
    else:
        formatted_location_sentences = [
            l[0][2:] if l[0][0] == ',' else l[0] for l in location_sentences]

        location_sentence = ', '.join(formatted_location_sentences[:-1])
        location_sentence += f' and {formatted_location_sentences[-1]}'

        formatted_location_sentences = [
            l[1][2:] if l[1][0] == ',' else l[1] for l in location_sentences]
        location_sentence_adv = ', '.join(formatted_location_sentences[:-1])
        location_sentence_adv += f' and {formatted_location_sentences[-1]}'

        return location_sentence, location_sentence_adv

def _get_time_sentence(
    temporal_information: Dict[str, str]
    ) -> str:
    """
    Get the sentence containing the time information.

    Parameters
    ----------
    temporal_information : { str: str }
        The dictionary containing the time information of the cluster.

    Returns
    -------
    str
        The sentence containing the time information.
    """
    if 'from time' in temporal_information:
        from_time = temporal_information['from time']
        to_time = temporal_information['to time']
        return f'from {from_time} to {to_time}'
    else:
        on_time = temporal_information['on time']
        return f'at {on_time}'

def _get_target_day_sentence(
    target_temporal_information: Dict[str, Any],
    ) -> str:
    """
    Get the sentence containing the day information of the cluster.

    Parameters
    ----------
    target_temporal_information : { str: Any }
        The dictionary containing the time information of the cluster.

    Returns
    -------
    str
        The sentence containing the day information of the cluster.
    """
    if 'from day' in target_temporal_information:
        from_day = target_temporal_information['from day']
        to_day = target_temporal_information['to day']
        from_date = target_temporal_information['from date']
        to_date = target_temporal_information['to date']
        day_sentence = f'from {from_day}, {from_date} to {to_day}, {to_date}'
    else:
        on_day = target_temporal_information['on day']
        on_date = target_temporal_information['on date']
        day_sentence = f'on {on_day}, {on_date}'

    return day_sentence

def _get_formatted_day_sentence(
    input_temporal_information: Dict[str, str],
    target_date_dt: datetime,
    is_target_more_days: bool,
    ) -> str:
    """
    Get the sentence containing the day information of the cluster
    formatted in a way that is more readable.

    Parameters
    ----------
    input_temporal_information : { str: str }
        The dictionary containing the time information of the cluster.
    target_date_dt : datetime
        The datetime of the date of the target cluster.
    is_target_more_days : bool
        Whether the target cluster spans in more days.

    Returns
    -------
    str
        The sentence containing the day information of the cluster
        formatted in a way that is more readable.
    """
    target_adjective = 'first' if is_target_more_days else 'same'

    # Case where the input temporal information spawns in more days.
    if 'from day' in input_temporal_information:
        input_from_date = input_temporal_information['from date']
        input_from_date_dt = datetime.strptime(input_from_date, '%d/%m/%Y')

        input_to_date = input_temporal_information['to date']
        input_to_date_dt = datetime.strptime(input_to_date, '%d/%m/%Y')

        input_from_day = input_temporal_information['from day']
        input_to_day = input_temporal_information['to day']

        # If input end date is the same as the target date.
        if input_to_date_dt == target_date_dt:
            # If input start date is the day before the target date.
            if input_from_date_dt == target_date_dt - timedelta(days=1):
                return f'from the previous to the {target_adjective} day'
            else:
                return f'from {input_from_day}, {input_from_date} to the {target_adjective} day'
        # If input end date is different from the target date.
        else:
            return f'from {input_from_day}, {input_from_date} to {input_to_day}, {input_to_date}'
    # Case where the input temporal information spawns in one day.
    else:
        input_on_date = input_temporal_information['on date']
        input_on_date_dt = datetime.strptime(input_on_date, '%d/%m/%Y')

        input_on_day = input_temporal_information['on day']
        # If input date is the same as the target date.
        if input_on_date_dt == target_date_dt:
            if is_target_more_days:
                return f'on the {target_adjective} day'
            else:
                return ''
        # If input date is the day before the target date.
        elif input_on_date_dt == target_date_dt - timedelta(days=1):
            return 'on the previous day'
        # If input date is different from the target date.
        else:
            return f'on {input_on_day}, {input_on_date}'

def _get_input_day_sentence(
    target_temporal_information: Dict[str, Any],
    input_temporal_information: Dict[str, Any],
    ) -> str:
    """
    Get the sentence containing the day information of the cluster.

    Parameters
    ----------
    target_temporal_information : { str: Any }
        The dictionary containing the time information of the target cluster.
    input_temporal_information :  { str: Any }
        The dictionary containing the time information of the input cluster.

    Returns
    -------
    str
        The sentence containing the day information of the cluster.
    """
    # Case where the target temporal information spawns in more days.
    if 'from day' in target_temporal_information:
        target_from_date = target_temporal_information['from date']
        target_from_date_dt = datetime.strptime(target_from_date, '%d/%m/%Y')
        return _get_formatted_day_sentence(
            input_temporal_information,
            target_from_date_dt,
            is_target_more_days=True)

    # Case where the target temporal information spawns in one day.
    else:
        target_on_date = target_temporal_information['on date']
        target_on_date_dt = datetime.strptime(target_on_date, '%d/%m/%Y')
        return _get_formatted_day_sentence(
            input_temporal_information,
            target_on_date_dt,
            is_target_more_days=False)
