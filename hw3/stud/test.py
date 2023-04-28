from stud.implementation import build_model_123, build_model_23, build_model_3


model = build_model_3("gpu")


list = [{'id': 'validation-449', 'text': "The iron-hulled steamer Gertrude was built in Whiteinch, Glasgow, Scotland as Yard No.100 at the Clydeholm yard of Barclay, Curle & Company as an American Civil War blockade runner and launched on 25 November 1862. Along with her sistership Emma, she was built for Thomas Stirling Begbie, a London shipowner and merchant.",'pron': 'her', 'p_offset': 226, 'entity_A': 'Gertrude', 'offset_A': 24, 'entity_B': 
'Emma', 'offset_B': 241},
{'id': 'validation-450', 'text': "He then agrees to name the gargoyle Goldie, after ``an old friend'', though he tells Goldie in private that he will continue to think of him as Irving. Goldie appears for a short scene in The Doll's House in which he is sitting upon Abel's shoulder as Lucien asks Abel about the inhabitants of the house. He later appears throughout the ``Parliament of Rooks'' story in Fables and Reflections, and briefly at the beginning of Brief Lives.", 'pron': 'He', 'p_offset': 305, 'entity_A': 'Lucien', 'offset_A': 252, 'entity_B': 'Abel', 'offset_B': 264}]


model.predict(list)
