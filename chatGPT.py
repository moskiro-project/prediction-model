import openai
import pandas as pd

openai.api_key =

def chatWithGPT(prompt):
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
  {"role": "user", "content": prompt}
  ]
  )
  return print(completion.choices[0].message.content)

def getJobTitle(row):
  #make prompt to determine job title
  prompt = "Benenne den Berufstitel für die folgende Beschreibung mit einem Wort: " + row[0] + ": " + row[1]
  #only keep first word of response
  #response = chatWithGPT(prompt).split()[0].lower()
  response = "Ingenieur"
  return response.lower()
  
 def getJobTitle(row):
  #make prompt to determine job title
  prompt = "Benenne den Berufstitel für die folgende Beschreibung mit einem Wort: " + row[0] + ": " + row[1]
  #only keep first word of response
  #response = chatWithGPT(prompt).split()[0].lower()
  response = "Ingenieur"
  return response.lower()

def getSkills(row):
  formatting_example = ""
  prompt = "Filtere Fähigkeiten und Qualifikationen aus dem folgenden Jobprofil: \"" + row[1] + "\"; und befolge dabei dieses Beispiel: " + formatting_example
  # make list
  #response = chatWithGPT(prompt).split().apply(lambda x: x.lower())
  response = ["Technisches Zeichnen", "Systementwurf"]
  return response
  
 def process_row(row):
  # Perform some processing on the row
  processed_row = row
  processed_row[3] = row[0]
  processed_row[0] = getJobTitle(row)
  return processed_row

# Read the input Excel file
input_file = 'MainDatensatz_5067Einträge_LN'
df = pd.read_excel(input_file)

# Apply the process_row function to each row
processed_rows = df.apply(process_row, axis=1)

# Write the processed rows to a new Excel file
output_file = 'output.xlsx'
processed_rows.to_excel(output_file, index=False)