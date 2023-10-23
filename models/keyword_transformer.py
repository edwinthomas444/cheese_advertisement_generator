from happytransformer import HappyGeneration, GENSettings

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-1.3B")


training_cases = """Keywords: persuasive advirtisement,<NameOfCheese:Grandma Singletons Beacon Fell PDO Traditional Creamy Lancashire><QualityOfRind:Creamy><AnimalSource:None><MainIngredient:None><MainIngredientWeight:None><NameOfCheesery:Grandma Singletons><NamesOfFacilities:None><ManufacturePlace:Lancashire><ManufacturedSinceDate:None><CheeseOriginatedDate:None><CheeseCreatorMovedFromPlace:None><CheeseCreatorMovedToPlace:None><CheeseNameFromObject:Beacon Fell>
Output: Grandma Singletons Beacon Fell PDO Traditional Creamy Lancashire is a true gem of Lancashire, produced by the renowned cheese manufacturer Grandma Singletons. This cheese has a rich history and geographical provenance, as it is made using locally sourced milk from farms within a 12-mile radius.
###
Keywords: persuasive advirtisement,<NameOfCheese:Bonchester><QualityOfRind:artisan><AnimalSource:None><MainIngredient:None><MainIngredientWeight:None><NameOfCheesery:Easter Weens Farm><NamesOfFacilities:None><ManufacturePlace:Bonchester Bridge, Roxburghshire><ManufacturedSinceDate:1980><CheeseOriginatedDate:None><CheeseCreatorMovedFromPlace:None><CheeseCreatorMovedToPlace:None><CheeseNameFromObject:None>
Output: Introducing Bonchester, a delectable British cheese that has been awarded with a Protected Designation of Origin (PDO). This artisan cheese is made in the border lands of England and Scotland, specifically in Bonchester Bridge, Roxburghshire. It was first developed in 1980 on the Easter Weens Farm and has since gained recognition for its exceptional quality. 
###"""

def create_prompt(training_cases, keywords):
  keywords_string = ", ".join(keywords)
  prompt = training_cases + "\nKeywords: "+ keywords_string + "\nOutput:"
  return prompt


keywords = ['persuasive advirtisement','<NameOfCheese:Buxton Blue><QualityOfRind:modern creamery><AnimalSource:cow><MainIngredient:None><MainIngredientWeight:None><NameOfCheesery:None><NamesOfFacilities:None><ManufacturePlace:Buxton, Derbyshire, England><ManufacturedSinceDate:None><CheeseOriginatedDate:None><CheeseCreatorMovedFromPlace:None><CheeseCreatorMovedToPlace:None><CheeseNameFromObject:Blue Stilton>']
prompt = create_prompt(training_cases, keywords)


args_beam = GENSettings(num_beams=5, no_repeat_ngram_size=3, early_stopping=True, min_length=1, max_length=150)

result = happy_gen.generate_text(prompt, args=args_beam)
print(result)

'''Buxton Blue, a modern creamery blue cheese, 
is a cousin of the famous Blue Stilton.
It is made from cow's milk and has a Protected
 Designation of Origin (PDO), 
 ensuring its authenticity and quality. 
 Originating from Buxton, 
 Derbyshire in England, this cheese has a rich history 
 and is a proud representation of the United Kingdom's 
 cheese-making tradition.'''


'''
<NameOfCheese:Buxton Blue>
<QualityOfRind:modern creamery>
<AnimalSource:cow>
<MainIngredient:None>
<MainIngredientWeight:None>
<NameOfCheesery:None>
<NamesOfFacilities:None>
<ManufacturePlace:Buxton, Derbyshire, England>
<ManufacturedSinceDate:None>
<CheeseOriginatedDate:None>
<CheeseCreatorMovedFromPlace:None>
<CheeseCreatorMovedToPlace:None>
<CheeseNameFromObject:Blue Stilton>
'''

# generated text
'''
Buxton Blue Stilton PDO Modern Creamery 
is a modern creamery based in Buxton, England. 
This creamery was established in the early 1990s 
and is one of the oldest creameries in the UK. 
The creamery produces a wide range of dairy products, 
including milk, cream, butter, cheese, yoghurt and ice cream. 
It is also the only creamery in the United Kingdom 
to be awarded a PDO for its creamery products. 
The PDO was granted to the creamery 
on the basis that the products 
are produced in accordance with 
the highest standards of quality 
and are of a high standard of excellence.
'''