import torch
import numpy as np
import sys
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import spacy

import LinkPred


example_text = """Unternehmensbeschreibung
  Securitas ist größter Anbieter professioneller Sicherheitslösungen mit 21.000 Mitarbeitenden in Deutschland und beschäftigt weltweit 358.000 Mitarbeiter in 45 Ländern. Mit unseren Werten Ehrlichkeit, Aufmerksamkeit und Hilfsbereitschaft ist es unser Ziel, die Welt sicherer zu machen.
  Securitas Electronic Security ist Teil der Securitas Gruppe und einer der führenden Anbieter von Sicherheitssystemen in Deutschland. Wir entwickeln innovative und zukunftsweisende Sicherheitskonzepte, bestehend aus maßgeschneiderten Videoüberwachungs-, Zutrittskontroll-, Einbruchmelde- sowie Brandmeldesystemen. Darüber hinaus übernehmen wir Alarmmanagement-Leistungen durch unsere eigenen Notruf- und Serviceleitstellen. Mit 18 Service- und Vertriebsstandorten ist Securitas Electronic Security flächendeckend präsent. Über 500 Mitarbeiter, davon nahe 300 Techniker, sorgen für die Sicherheit unserer Kunden in Deutschland.
 
  Montagetechniker / Servicetechniker (m/w/d) Sicherheitstechnik
  Hamburg, Deutschland
  Vollzeit
 
  Stellenbeschreibung
 
   Durchführung von Wartungen und Instandsetzungen an Sicherheitsanlagen und Sicherheitssystemen
   Inbetriebnahme von Sicherheitsanlagen und Sicherheitstechnik
   Erstellung von technischen Unterlagen samt Dokumentation
   Fehler- und Störungsbeseitigung an Sicherheitsanlagen und Sicherheitssystemen
 
  Qualifikationen
 
   Abgeschlossene Ausbildung im Bereich Elektrotechnik oder Handwerk
   Abgeschlossene Ausbildung als Elektroniker für Energie und Gebäudetechnik, Informationselektroniker, Elektroniker für Gebäude- und Infrastruktursysteme, Elektroniker für Betriebstechnik, Mechatroniker oder IT-Systemelektroniker
   Idealerweise bereits Berufserfahrung als Servicetechniker und Kenntnisse in einem Bereich der Sicherheitstechnik (Einbruchmeldeanlagen, Brandmeldeanlagen Videotechnik oder Zutrittskontrollen)
   Gutes technisches Verständnis
   Hohe Service- und Kundenorientierung
 
  Zusätzliche Informationen
  Was bieten wir
 
   Krisensicherer Arbeitsplatz
   Attraktives Gehaltspaket plus verschiedene Zusatzleistungen
   30 Tage Urlaub
   Neutraler Firmenwagen zur privaten Nutzung
   Persönliche Entwicklungsmöglichkeiten innerhalb von Securitas
   Schulungen und Trainings an der Securitas Akademie
   Strukturierte Einarbeitung
   Wertschätzende Unternehmenskultur
   Moderne Arbeitsmittel wie hochwertiges Werkzeug und Arbeitskleidung
   Exklusive Mitarbeiterangebote über Corporate Benefits
   Betriebliche Altersvorsorge (BAV)
   Kostenlose Securitas-Gruppenunfallversicherung – auch in der Freizeit
 
 
  Ist das Interesse geweckt?
  Klicken Sie direkt auf "Jetzt bewerben" und senden Sie uns ganz schnell und einfach Ihre Unterlagen zu.
  Bei Fragen stehen wir Ihnen Montags - Freitags von 08:00 - 17:00 Uhr telefonisch unter +49 800 73 27 848 gerne zur Verfügung. Wir freuen uns darauf.  Securitas steht für Chancengleichheit und Diversität! Wir lehnen Diskriminierung jeglicher Art ab und freuen uns auf Bewerbungen und die Zusammenarbeit mit Menschen – unabhängig von Geschlecht, sexueller Orientierung, ethnischer und sozialer Herkunft, Alter, Behinderung, Religion oder Weltanschauung
  Hinweis: Sie erklären sich mit Ihrer Einreichung der Bewerbung bereit, dass Ihre Bewerbung innerhalb der Securitas-Gruppe Deutschland und Kooperationspartner weitergeleitet wird. Dadurch erhöhen sich Ihre Bewerbungschancen. Wenn Sie dies nicht wünschen, teilen Sie uns dies bitte in Ihrem Schreiben mit.
  """


def extract_entities(text):
    nlp_ner = spacy.load('ner_model')
    doc = nlp_ner(text)
    
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    return entities

def main(text = "", doc2vec = True):

    print(example_text)
    input()
    result = extract_entities(example_text)
    print(result)
    input()
    #example result for testing
    #result = ["statistics", "feature_engineering", "scala", "aws"]
    #pred_model = LinkPred.model(load_model=True,save_model=False,load_test=False)
    if doc2vec :
        model = Doc2Vec.load("model/doc2vec_newData")
    else: model = Word2Vec.load("model/word2vec_newData")
    embedding = model.infer_vector(result)
    print(embedding)
    input()
    #embedding = embedding.reshape((1, 128))
    #print(embedding.shape)
    #pred_model.x = torch.cat((pred_model.x, torch.from_numpy(embedding)))
    #pred_model.y = np.asarray([0])
    #final_prediction = pred_model.test(topk=3)
    
    final_prediction = model.dv.most_similar([embedding], topn = 3)
    print(final_prediction)
    final_prediction = [job for job, similarity in final_prediction]
    print("Final Prediction: " + str(final_prediction))
    input()


if __name__ == '__main__':
    main()
    #if(len(sys.argv) > 0):
    #    main(sys.argv[0][0])
