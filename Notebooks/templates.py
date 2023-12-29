from string import Template

# prompt to be used for realligning pretraining text in specific rhetoricla structure for cheese descriptions
prompt_reallignment = Template('''For the given text '$text'
                        Rearrange the text to write pursuasive advirtisement cheese descriptions (of maximum 512 tokens) without excluding any details \
                        that follow a specific structure with 6 headings and a single paragraph below each heading without bullet points:
                        1) Description of Cheese (name, manufacturer, historical and geographical provenance)
                        2) Description of Product (characteristics (shape, size, texture, coating, weight), and ingredients (type of milk, rennet used))
                        3) Description of the Process (preparation and aging)
                        4) Description of Smell and Taste
                        5) Serving Suggesions (temperature, presentation, food-pairing, wine-pairing)
                        6) Quality Assurance (quotations, awards, quality tests)''')

# prompt to be used for slot value generation
prompt_slot_filling = Template('''Fill the slots '$slots' with words/phrases only from the text if present else specify 'None': 

$text''')


# 2 step pipeline
# Step1: prompt for generating values for given slots
# Step2: prompt for generating a text using all the given slots
prompt_slot_to_desc = [
    Template(
        "Return a JSON by filling the below slots with values: '$slot_descriptions_string'"),
    Template('''Without excluding any of the below slots generate a persuasive, professional and concise single paragraph text describing the process of cheese making: 
    '$slot_value_json'
    ''')
]
