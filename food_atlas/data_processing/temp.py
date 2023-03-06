import sys
import inflect

p = inflect.engine()


def singularize(input_str: str):
    output_str = []
    for x in input_str.split():
        x_singular = p.singular_noun(x)
        if x_singular is False:
            output_str.append(x)
        else:
            output_str.append(x_singular)

    return ' '.join(output_str)


result = singularize('cumarins')
print(result)
