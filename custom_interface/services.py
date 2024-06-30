import re

from custom_interface.core.enums import NumberCountryTypes


def kg_correctly(number_text: str):
    multiline_new_pattern = r"D\d{4}[A-Z]{1,3}"
    multiline_old_pattern = r"\d{4}[A-Z]{2,3}"
    if re.fullmatch(multiline_new_pattern, number_text):
        return number_text.replace('D', '0', 1)
    if re.fullmatch(multiline_old_pattern, number_text):
        return number_text[4] + number_text[0:4] + number_text[5:]
    return number_text


def am_correctly(number_text: str):
    multiline_pattern = r"\d{2}[A-Z]{2}AM\d{3}"
    if re.fullmatch(multiline_pattern, number_text):
        return number_text[0:4] + number_text[6:]
    return number_text
    


def adjustment_number_text(number_text: str, number_country: str) -> str:
    if number_country == NumberCountryTypes.kg.value:
        return kg_correctly(number_text)
    if number_country == NumberCountryTypes.am.value:
        return am_correctly(number_text)
    return number_text
