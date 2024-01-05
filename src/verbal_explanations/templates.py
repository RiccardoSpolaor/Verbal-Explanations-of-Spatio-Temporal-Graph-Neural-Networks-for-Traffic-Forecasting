
prediction_verbs = [
    'predicted',
    'anticipated',
    'forecasted',
    'expected']

first_paragraph_sentences = [
    'A {c} was {prediction} {w:} {d}, with an average speed of {s} km/h {t}.',
    '{D}, {w} was {prediction} to experience {c}, averaging {s} km/h {t}.',
    '{W} was {prediction} to {c} {d}, with an average speed of {s} km/h {t}.',
    'A {c} was {prediction} to hit {w} {d}, {t}, with an average speed of {s} km/h.',
    'A {c} was {prediction} on {w:} {d}, with an average speed of {s} km/h {t}.',
    '{W} was {prediction} to see {c} {d}, with an average speed of {s} km/h {t}.',
    'A {c} was {prediction} {w:} {d}, {t}, with an average speed of {s} km/h.',
    'A {c} was {prediction} to occur {w:} {d}, {t}, with an average speed of {s} km/h.',
    '{D}, a {prediction} {c} affected {w}, maintaining an average speed of {s} km/h {t}.',
    '{W} {prediction} {c} {d}, {t}, with an average speed of {s} km/h.']

extra_involved_street_sentences = [
    'The {c} also affected{w}.',
    'The {c} also impacted{w}.',
    'The {c} also hit{w}.',
    'The {c} also took place {w:}.',
    'The {c} also happened {w:}.',
    'The {c} extended {w:}.']

first_paragraph_end_sentences = [
    'This was caused by a',
    'This resulted from a',
    'This was induced by a',
    'This happened because of a',
    'This was driven by a',
    'This was a result of a',
    'The motivation was a',
    'This was triggered by a',
    'This occurred because of a',
    'The reason behind it was a']

second_paragraph_connectors = [
    'An initial {c}',
    'A first {c}',
    'Firstly, a {c}',
    'Initially, a {c}',
    'To begin, a {c}',
    'To start, a {c}',
    'To commence, a {c}',
]

second_paragraph_verbs = [
    'occurred',
    'happened',
    'manifested',
    'materialized',
    'took place'
]

other_paragraphs_connectors = [
    'Following this, {c}',
    'Subsequently, {c}',
    'Next, {c}',
    'Then, {c}',
    'Afterwards, {c}',
    'After this, {c}',
    'After that, {c}',
]

final_paragraph_connectors = [
    'Finally, {c}',
    'Lastly, {c}',
    'Eventually, {c}',
    'To conclude, {c}',
    'In the end, {c}',
    'Ultimately, {c}',
    'At last, {c}']

second_paragraph_sentences = [
    ' {w:} {t} {d}, with an average speed of {s} km/h.',
    ' {w:}, occurring {t} {d}, with an average speed of {s} km/h.',
    ', averaging at a speed of {s} km/h, {w:} {d}, {t}.',
    ', at {s} km/h, {w:} {d}, {t}.',
    ', at {s} km/h, {w:} {t} {d}.',
    ', with an average speed of {s} km/h, {w:} occurring {t} {d}.',
    ' {w:}, with an average speed of {s} km/h, {d}, {t}.',
    ' {w:}, occurring {t} {d}, with an average speed of {s} km/h.',
    ' {w:}, with an average speed of {s} km/h, {d}, {t}.',
    ' {d}, {t} {w:} with an average speed of {s} km/h.']

another_connectors = [
    'another',
    'a new',
    'a further',
    'an additional',
    'an extra']

again_connectors= [
    'again',
    'once more',
    'another time']

yet_again_connectors = [
    'yet again',
    'once again',
    'yet another time']