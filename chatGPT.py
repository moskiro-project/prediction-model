import openai
import pandas as pd

openai.api_key = "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"

# This file is not supposed to be in this repo. Check GPTAPI repo if this can be deleted or should be moved.

example_text = """Benenne den Berufstitel für die folgende Beschreibung mit einem Wort: Metallbauer (m/w/d): Das Handwerksunternehmen der Zukunft.
  Sie sind ein Teamplayer? Sie haben Anspruch – an sich selbst und uns? Dann kommen Sie zur HWP Handwerkspartner AG  als Feuerungstechniker (m/w/d)  an unseren Standort GTB Berlin Kundendienst GmbH in Berlin.  GTB Kundendienst Berlin - immer für Sie da. Wir wollen Sie als Kunden zufriedenzustellen, deswegen steht ein guter Service für uns stets an erster Stelle. Egal, ob für ein Beratungsgespräch, einen Wartungstermin oder im Notfall: Der GTB Berlin Kundendienst ist immer für Sie da, wenn Sie ihn brauchen – online, telefonisch und vor Ort.
  Darauf können Sie sich freuen:
 
 
   Eine wachsende und solide Unternehmensgruppe mit 15 Standorten
   1000 Kollegen aus dem Handwerk
   Eine moderne Arbeitswelt
   Einen unbefristeten Arbeitsvertrag
   Fairer Stundenlohn und Weihnachtsgeld
   Betriebliche Altersvorsorge
   Kindergartenzuschuss steuerfrei, mtl. bis zu 150€ / Kind
   Entscheidungsfreiräume & ein angenehm kollegiales Team
   Professionelle Werkzeuge & Materialien
   Hochwertige Arbeitskleidung
   Berufliche Entwicklungsmöglichkeiten & Weiterbildungsangebote
   Eine offene Kommunikation
   Für den Spaß - regelmäßige Mitarbeiter-Events
   Stets gut informiert – Mitarbeiterzeitung „HWP Echo!“
 
 
  Über diese Qualifikationen freuen wir uns:
  Wir suchen kompetente Kollegen mit Freude am Job – in unserem Team finden sich die unterschiedlichsten Erfahrungen, Kenntnisse und Charaktere wieder. Und davon wünschen wir uns noch mehr.
 
   Abgeschlossene Ausbildung zum Anlagenmechaniker Sanitär-, Heizung- und Klimatechnik oder eine ähnliche Qualifikation
   Berufserfahrung mit Feuerungsanlagen
   Sicherheitsbewusstsein im Umgang mit Gefahrenstoffen
   Sie sind gewissenhaft, akkurat und teamfähig
   Sie haben Lust sich aktiv einzubringen
   Freude am Beruf ist Ihnen wichtig
   Freundlicher Umgang mit Kunden ist für Sie selbstverständlich
   Sie haben einen gültigen Führerschein der Klasse B
 
 
  Hier bringen Sie sich ein:
 
 
   Durchführung von Inbetriebnahme- und Servicearbeiten
   Fehlersuche an Anlagen von Kunden inkl. Dokumentation mit digitaler Unterstützung
   Pflege der Feuerungsanlagen sowie deren Zubehör
   Prüfen der Schutzeinrichtungen
   Wartung und Instandhaltung von Feuerungsanlagen, Pumpen und weiteren gebäudetechnischer Anlagen und deren Komponenten
   Umbau und Reparatur bestehender Anlagen
   Montage von Neuanlagen
 
 
  Neugierig auf uns?
  Wir freuen uns auf Ihre Bewerbung!
  Kontaktinformationen
 
 
   Den "Jetzt-bewerben-Button" drücken
   Alternativ per Mail an karriere@handwerkspartner.de senden
  Ihr Ansprechpartner ist Rainer Heins vom HWP Personalmanagement.
 
  Über uns
 
  Wir, die HWP Handwerkspartner AG, sind ein junges, dynamisch wachsendes Handwerksunternehmen, das 2007 gegründet wurde. Die Gruppe ist mit 15 Standorten in Deutschland und Luxemburg präsent. Eine weitsichtige Unternehmenspolitik, ein umfangreiches Prozess- und Qualitätsmanagement sowie risikobewusstes Handeln und Kundenorientierung sind gelebte Unternehmenskultur. Eine ebenfalls entscheidende Rolle für unseren Erfolg sehen wir in der konsequenten Förderung und Weiterentwicklung unserer Mitarbeiter."""

NER_example = """Die iw-projekt GmbH ist ein international operierendes Engineering-Unternehmen, mit langjähriger Erfahrung in der Umsetzung diffiziler Maschinen- und Anlagenbauprojekte. Unser innovativer Geschäftsbereich Personaldienstleistung rekrutiert gefragte High Professionals, Ingenieure, Techniker und Facharbeiter, für interessante Kunden aus nahezu allen technischen Bereichen. Ein rundum faires Miteinander, maßgeschneiderte Coachings und eine ganzheitliche Betreuung im Prozess bringen Sie ohne Umwege zu Ihrem Traumjob!  Ab sofort suchen wir einen 
     Industriemechaniker, Fertigungsmechaniker für Montage/Service (m/w/d) ID: 21954 
     Was wir Ihnen bieten:  Unser hoch motiviertes Team steht Ihnen mit Rat und Tat zur Seite. Wir akquirieren Vorstellungstermine in dem von Ihnen definierten Umfeld und unterstützen Sie als Ihr persönlicher Dienstleister, bis Sie Ihre Wunschposition gefunden haben! Selbstverständlich mit unbefristetem Arbeitsvertrag sowie einer leistungs- und marktgerechten Vergütung. 
     Wir zahlen Ihnen eine EINSTELLUNGSPRÄMIE in Höhe von 1.000,00 € . 500,00 € mit dem ersten Lohn und weitere 500,00 € nach Ablauf der Probezeit.  Ihre Aufgaben: 
     
      Eigenständiges Arbeiten nach Technische Unterlagen, wie z.B. Konstruktionszeichnungen, Fertigungs-, Montagepläne und Stücklisten. 
      Erfahrung in der Montage von Einzelteilen und Gesamtbaugruppen 
      Erfahrung in den Bereichen Baugruppen einstellen und auf Funktion prüfen. 
      Erfahrung im Bereich der Fügetechnik. z.B. Kleben, Schraub- und Stiftverbindungen (Kegelstift) usw. 
      Sicherer Umgang mit Mess- und Prüfwerkzeugen. 
      Endkontrollen / Abnahme von fertiggestellten Baugruppen. 
      
     Was wir von Ihnen erwarten: 
     
      Abgeschlossene Berufsausbildung als Industrie- oder Fertigungsmechaniker 
      Einsteiger willkommen 
      Ausgeprägtes technisches Verständnis und handwerkliches Geschick 
      Selbstständige und sorgfältige Arbeitsweise 
      Freude an der Arbeit im Team 
      Englischkenntnisse wünschenswert 
      Reisebereitschaft
"
gefiltertes Resultat: 
technisches Verständnis
handwerkliches Geschick
Englisch
Berufsausbildung Fertigungsmechaniker"""


def chatWithGPT(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


def getJobTitle():
    # make prompt to determine job title
    # prompt = "Benenne den Berufstitel für die folgende Beschreibung mit einem Wort: " + row[0] + ": " + row[1]
    # print(prompt)
    prompt = example_text
    # only keep first word of response
    response = chatWithGPT(prompt)
    if response is None:
        response = ""
    return response


def getSkills(row):
    prompt = "Filtere einzelne Qualifikationen aus dem folgenden Jobprofil als Liste ohne Kommentar ohne Wörter zu ändern: \"" + \
             row[1] + "\"; und befolge dabei dieses Beispiel: " + NER_example
    # make list
    response = chatWithGPT(prompt).split()
    # response = ["Technisches Zeichnen", "Systementwurf"]
    return response


def process_row(row,skills = True):
    # Perform some processing on the row
    processed_row = row
    processed_row[3] = row[0]
    if skills:
        processed_row[1] = getSkills(row)
    else :
        processed_row[0] = getJobTitle(row)

    return processed_row

