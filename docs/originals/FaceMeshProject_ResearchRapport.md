
MeshExpress
Facemesh uitdrukkingsherkenning
Onderzoeksrapport
![](data:image/png;base64...)
**Colofon**
| **Auteur** | **Studentnummer** |
| --- | --- |
| Kevin van Hoeijen | 2118888 |
| Enes Çaliskan | 1671261 |
| William Hak | 1659237 |
| Jaap-Jan Groenendijk | 1548148 |
| **Versie** | **Wijziging** | **Datum** |
| --- | --- | --- |
| 0.1 | Initele opzet | 23-09-2024 |
| 0.2 | Aanvullen hoofdstukken 1, 2, 3 en 4 | 08-10-2024 |
| 0.3 | Finaliseren hoofdstukken 1, 2, 3 en 4. Begonnen met hoofdstukken 5 en 7 | 17-10-2024 |
| 0.4 | Finaliseren hoofdstuk 5, begonnen met hoofdstuk 6, 7 en 8 | 22-10-2024 |
| 0.5 | Finaliseren alle hoofdstukken, reviewen van alle hoofdstukken | 01-11-2024 |
| 0.6 | Hoofdstukken gereviewd en aangepast waar nodig | 04-11-2024 |
| 0.7 | Alles afronden en gehele document finaliseren | 05-11-2024 |
| 1.0 | Final versie van het document | 06-11-2024 |
Datum van laatste update: 14-01-2025
Samenvatting
Om een gezichtsuitdrukking herkenningsmodel te trainen, is een set met coördinaten van gezichtsfeatures benodigd. Om deze coördinaten te extraheren uit een foto van een gezicht is Google Facemesh beschikbaar gesteld. Dit resulteert in een set met coördinaten van features zoals wenkbrauwen, mondhoeken etc.
Om het model te trainen zodat het geschikt is voor een diversiteit aan personen, moet het ook getraind worden met een grote diversiteit aan foto’s. Om dit te bewerkstelligen zijn er twee opties, een natuurlijke dataset vergaren, of een synthetische dataset maken door middel van een al bestaand generatie model om een grote diversiteit aan gezichten te creëren. We hebben ervoor gekozen om beide methoden te gaan uitvoeren. Het handmatig samenstellen van de dataset is beperkt omdat dit veel tijd vergt en ethische vragen met zich meebrengt. Om het synthetische dataset te maken blijkt StyleGAN een geschikte kandidaat.
De input data voor het herkenningsmodel is relatief groot, omdat het model schaalbaar moet zijn om de diversiteit te waarborgen, valt de decision tree af. Omdat de Facemesh een grote set aan features extraheert en deze een complex verband tussen features heeft is Lineaire regressie ook niet geschikt.
Verder lijken K-Means clustering, support vector machine en Autoencoders de meest geschikte algoritmen omdat ze schaalbaar en geschikt voor grote datasets zijn, welk algoritme optimaal is moet middels testen worden gevonden.
Om de datasets te vergaren zijn een aantal ethische punten belangrijk. De privacy van mensen moet gewaarborgd blijven, wat betekend dat foto’s alleen mogen verzameld worden wanneer de personen daar toestemming toe geven. Dit ethische punt is ook van belang bij het gebruikt van het model wanneer het getraind is en gaat classificeren. Door bijvoorbeeld beelden die gemaakt worden om te classificeren alleen lokaal op te slaan tot de classificatie is gedaan en daarna de beelden te verwijderen kan je deze privacy ook waarborgen. Verder is het belangrijk om transparant te zijn naar gebruikers over hoe data wordt gebruikt. Een groot voordeel van het synthetische dataset is dat de gezichten gegenereerd worden. Dit betekend dat de mensen niet bestaan en er dus ook geen toestemming nodig is om privacy te waarborgen. Dit scheelt een hoop tijd in het vergaren van de dataset.
Inhoudsopgave
[1. Inleiding 5](#_Toc181788595)
[1.1. Projectomschrijving 5](#_Toc181788596)
[1.2. Stakeholders 5](#_Toc181788597)
[2. Onderzoeksvragen 6](#_Toc181788598)
[2.1. Hoofdvraag 6](#_Toc181788599)
[2.2. Deelvragen 6](#_Toc181788600)
[3. Deelvraag 1 – Emotie classificatie 7](#_Toc181788601)
[4. Deelvraag 2 – Synthetische data 8](#_Toc181788602)
[4.1. Bestaande natuurlijke datasets 8](#_Toc181788603)
[4.1.1. Synthetische datasets 9](#_Toc181788604)
[4.1.2. Zelfgemaakte – niet synthetische dataset 14](#_Toc181788605)
[4.1.3. Conclusie 14](#_Toc181788606)
[5. Deelvraag 3 – Computer vision 15](#_Toc181788607)
[5.1. Overview 15](#_Toc181788608)
[5.2. Feature extraction 15](#_Toc181788609)
[6. Deelvraag 4 – Algoritme voor usecase 17](#_Toc181788610)
[6.1. Support vector machine 17](#_Toc181788611)
[6.1.1. Lineaire SVM 17](#_Toc181788612)
[6.1.2. Niet lineaire SVM 17](#_Toc181788613)
[6.2. K-Means clustering 19](#_Toc181788614)
[6.3. Decision tree 20](#_Toc181788615)
[6.4. Lineaire regressie 21](#_Toc181788616)
[6.5. Autoencoders 22](#_Toc181788617)
[6.6. Conclusie 22](#_Toc181788618)
[7. Deelvraag 5 - Google facemesh 23](#_Toc181788619)
[7.1. Mediapipe Face mesh 24](#_Toc181788620)
[7.1.1. Face Mesh pipeline 24](#_Toc181788621)
[7.1.2. Gebruik Mediapipe Face Mesh 24](#_Toc181788622)
[7.2. Gelaatsdelen herkennen uit het gaas 24](#_Toc181788623)
[7.3. Conclusie 25](#_Toc181788624)
[8. Deelvraag 6 – Ethische overwegingen 26](#_Toc181788625)
[9. Bronnen 28](#_Toc181788626)
[10. Bijlagen 30](#_Toc181788627)
[10.1. Voorbeeldcode Google/Mediapipe facemesh 30](#_Toc181788628)
[10.1.1. Face\_landmark\_detection.py 30](#_Toc181788629)
[10.1.2. Face\_landmark\_visualization.py 31](#_Toc181788630)
[10.1.3. run\_facemark\_pipeline.py 32](#_Toc181788631)
# Inleiding
Het verkrijgen van goed getraind zorgpersoneel wordt steeds lastiger in de maatschappij. Hierdoor is het noodzakelijk om, wanneer gepast, deze taken over te kunnen dragen aan geautomatiseerde systemen zodat de werkdruk bij het al schaarse personeel gedrongen kan worden.
Door het inzetten van “gezelschap robots” kan er verdiepende zorg worden verleend aan kinderen die emotionele ondersteuning nodig hebben tijdens een medisch traject waarbij niet altijd een medewerker aanwezig kan zijn.
## Projectomschrijving
Het doel van het project is om een API te maken die informatie kan overdragen uit een camerabeeld naar een andere applicatie om de gezichtsuitdrukking van een persoon te kunnen verwerken.
Tijdens het onderzoek wordt onderzoek gedaan naar het verwerken van data door middel van een machine learning model en het vergaren van een dataset voor dit model om het te trainen en ingezet kan worden over de datastream van een camera.
Als laatste wordt geëvalueerd over de ethische vraagstellen die komen kijken bij het outsourcen van menselijk werk naar robots, videobeelden maken van mensen, en de veiligheid van kinderen.
## Stakeholders
Jeroen Veen, vraagt namens HAN University of Applied Sciences, aan de studenten benoemd op de voorpagina om een extensie te ontwikkelen op Google Face Mesh. Google wordt gezien als de product owner van Google Face mesh. De projectgroep staat niet in contact met Google over de ontwikkeling van de Face Mesh extensie.
# Onderzoeksvragen
Om de onderzoeksvraag eenvoudiger te beantwoorden, wordt de vraag uiteengezet in meerdere deelvragen. Aan de hand van de antwoorden op de deelvragen, zal de hoofdvraag beantwoord worden.
## Hoofdvraag
### De hoofdvraag op het onderzoek is:

**“Hoe kunnen gezichtsuitdrukkingen gedetecteerd en herkend worden met een camera?”**
## Deelvragen
Het onderzoek is onderverdeeld in de volgende onderzoeksvragen en zullen in deze volgorde behandeld worden in onderzoeksresultaten.
1. **Op welke manier zijn glimlachen, hard lachen en fronzen van elkaar te onderscheiden?**Deze vraag zal vertellen met welke kenmerken er onderscheid valt te maken tussen deze verschillende gezichtsuitdrukkingen en ons in staat brengen grenzen te stellen aan de date die vergaard moet worden
2. **Is het mogelijk om natuurlijke en synthetische datasets te vergaren die bruikbaar om ons algoritme te trainen?**Door onderzoek te doen naar het verschil in gebruik van natuurlijke en synthetische beelden brengen we ons in staat om een mogelijk geautomatiseerd trainingsmodel te gebruiken.
3. Hoe zijn de gezichtsuitdrukkingen glimlachen, hard lachen, fronzen en neutrale blik van elkaar te onderscheiden via computer vision?
   Door onderzoek te doen naar het extraheren van features d.m.v. computer vision kunnen we labels maken om labels te controleren. Bijvoorbeeld door thresholding om een achtergrond weg te halen, afstand tussen veelzeggende gezichtskenmerkpunten etc.
4. **Welk algoritme kunnen we het beste gebruiken met de vergaarde data?**Door verschillende algoritmes met elkaar te vergelijken kan er een beslissing gemaakt worden over het best passende algoritme.
5. Hoe kunnen we Google Facemesh gebruiken om gezichtsuitdrukkingen te herkennen?
   Door onderzoek te doen naar Google Facemesh en het toepassen hiervan
6. Welke ethische overwegingen zijn genomen bij gezichtsuitdrukking detectie?
# Deelvraag 1 – Emotie classificatie
De American Psychological Association beschrijft een gezichtsuitdrukking als een manier van non-verbale communicatie waarbij gezichtsspieren bewegen. Het laat zien hoe iemand zich voelt en kan helpen om te overleven, maar het wordt ook beïnvloed door culturele regels over hoe je emoties toont en door genetische factoren. (Facial expression, 2018)
Om onderzoek te kunnen doen naar het herkennen van gezichtsuitdrukkingen, is het eerst noodzakelijk om vast te stellen hoe verschillende uitdrukkingen onderscheiden kunnen worden. Dit onderzoek richt zich specifiek op drie gezichtsuitdrukkingen: glimlachen, hardop lachen en fronsen. Voor menselijke begrippen, niet voor computers, definiëren we de gezichtsuitdrukkingen als volgt:
* **Glimlach**. De glimlach is een gezichtsuitdrukking waarbij de mondhoeken omhoog gaan. Vaak bewegen de spieren rond de ogen mee, waardoor kleine rimpeltjes ontstaan, vooral bij een oprechte glimlach. Deze natuurlijke beweging laat zien dat iemand blij of vriendelijk is. (Jaffe, 2011)
* **Glimlach met zichtbare tanden**. Ook wel een “tandige glimlach” genoemd, is wanneer de mondhoeken omhoog gaan en de bovenste tanden zichtbaar worden. Dit gebeurt door het samentrekken van bepaalde spieren: de zygomaticus major, die de mondhoeken optilt, en de levator labii superioris, die de bovenlip omhoog trekt. Hoeveel tanden te zien zijn, hangt af van iemands schedel structuur en hoe breed de glimlach is. (DDS, 2005)
* **Fronsen**. Een frons is een gezichtsuitdrukking waarbij de mondhoeken naar beneden gaan en de wenkbrauwen naar elkaar toe en naar beneden bewegen, wat vaak ontevredenheid, verdriet of concentratie uitdrukt. Deze uitdrukking wordt veroorzaakt door de werking van verschillende spieren: de corrugator supercilii trekt de wenkbrauwen naar beneden en naar elkaar toe, waardoor verticale rimpels tussen de wenkbrauwen ontstaan; de depressor anguli oris trekt de mondhoeken naar beneden en vormt de typische frons. Deze combinatie van spierbewegingen zorgt voor de bekende uitdrukking die negatieve emoties of diepe gedachten laat zien. (Sendic, 2023)
### De verschillen tussen deze emoties zijn als volgt vast te stellen:

| **Gezichtsuitdrukkingen** | **Verschil makende factor** |
| --- | --- |
| Glimlach vs glimlach met zichtbare tanden | Tanden zichtbaar? Dan spreken we over glimlachen met zichtbare tanden |
| Glimlach vs fronsen | Wenkbrauwen naar beneden gericht? Dit wordt herkend als fronsen |
| Glimlach met zichtbare tanden vs fronsen | Tanden zichtbaar? Dit wordt herkend als glimlach met zichtbare tanden. |
Tot dusver zijn de gezichtsuitdrukkingen uitgedrukt in menselijke termen. Voor dit onderzoeksproject is het belangrijk dat een digitaal algoritme onderscheid kan gaan maken tussen de genoemde emoties. Over emotie classificatie met computer vision is meer te lezen in Deelvraag 3 – Computer vision.
# Deelvraag 2 – Synthetische data
Het verzamelen van natuurlijke en synthetische datasets voor het trainen van ons machinelearning model is een complexe taak. Door bestaande datasets te gebruiken en onderzoek te doen naar synthetische datasets, kunnen we vaststellen of machine learning en synthetische beeldgeneratie gecombineerd kunnen worden voor het classificeren van gezichtsuitdrukkingen
## Bestaande natuurlijke datasets
In het verleden is veel onderzoek gedaan naar gezichtsherkenning, waardoor er veel datasets beschikbaar zijn op het internet. Het kan echter lastig zijn om de juiste datasets te vinden door het ruime aanbod van datasetaanbieders, research papers en GitHub-repositories. Een greep uit de allergrootste bestaande datasets:
* CASIA-Webface
* CelebA
* UMDFace
# * VGG2
# * M21M

* Glint360K
* IMDB-Face
(bron: https://github.com/deepinsight/insightface/blob/master/recognition/\_datasets\_/README.md)
#### Kaggle
Kaggle is een wereldwijd community-based platform waar meer dan 20 miljoen gebruikers, waaronder datawetenschappers, machinelearning-beginners en onderzoekers, bijdragen aan de ontwikkeling van machinelearning. Kaggle biedt een breed scala aan datasets, machinelearningmodellen en mogelijkheden voor stresstests van deze modellen.
Kaggle maakt het eenvoudig om toegang te krijgen tot een grote collectie datasets, wat data-acquisitie versnelt en vereenvoudigt. De datasets op Kaggle zijn vooraf geclassificeerd en soms ook gepreprocessed.
Kaggle organiseert ook competities waarin grote bedrijven zoals Google, Quora, Mercedes-Benz en TensorFlow grote geldprijzen uitreiken voor de beste oplossingen. Deze competities stimuleren kwalitatieve inzendingen van deelnemers.
### Bronnen grote bedrijven kaggle:

* <https://www.kaggle.com/competitions/gemini-long-context>
* [https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing\](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing%5C)
* <https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge>
##### Gezichtsherkenningsdatasets Kaggle
Hoewel Kaggle bijna 400.000 datasets biedt, blijkt het een uitdaging om een geschikte dataset te vinden voor onze doeleinden. Veel datasets zijn namelijk geclassificeerd volgens emoties als blij, boos, angstig, neutraal en verdrietig.
Aangezien de labels van deze afbeeldingen niet volledig aansluiten op onze toepassing, kunnen we door deze fotoarchieven te filteren, zelf selecteren welke afbeeldingen bruikbaar zijn en welke gezichtsuitdrukkingen ze representeren.
Dit proces zal echter tijdrovend en mogelijk zonder succes zijn.
### Synthetische datasets
Het genereren van gezichten is een groeiend veld in computer vision en AI, waarbij gezichten worden gegenereerd zonder menselijke betrokkenheid.
### #### GAN (Generative Adversarial Network):

Een GAN (Generative Adversarial Network) is momenteel een van de meest gebruikte methoden voor het genereren van gezichten. Een GAN bestaat uit een generator en een discriminator.
De generator maakt nieuwe data op basis van willekeurige nummers en creëert daarmee een afbeelding.
De discriminator controleert, op basis van publieke foto’s van beroemdheden, of het een echte of gegenereerde afbeelding is.
De generator blijft afbeeldingen maken totdat de discriminator de gegenereerde data als echt beschouwt.
![](data:image/png;base64...)
(Chauhan, 2024)
#### StyleGAN2
Een verder ontwikkelde open-source GAN voor gezichtscreatie is StyleGAN2 van NVIDIA. Het gebruik van StyleGAN2 heeft voordelen zoals de hoge beeldkwaliteit, het geen rekening hoeven houden met lichtval, hardware beperkingen van camera’s en het verzamelen van een diverse dataset.
![](data:image/png;base64...)
Het zelf instellen en trainen van een GAN biedt flexibiliteit en maakt het mogelijk een dataset te genereren die perfect aansluit op de behoeften van ons model.
Een nadeel is echter de kans op bias in de dataset door te genereren waar je naar zoekt, wat de nauwkeurigheid van ons trainingsmodel kan beïnvloeden.
Daarnaast kunnen hardware beperkingen het genereren van een uitgebreide dataset beperken.
# (NVIDIA, 2024)

Een (getrainde) GAN neemt een “seed” (een willekeurig getal) en genereerd aan de hand van dit getal een afbeelding.
#### Testen StyleGAN2
Het gebruik van StyleGAN2 en hier gepaste emoties blijkt echter een grote uitdaging te zijn. Door de complexiteit van het uitgebreide neurale netwerk is het instellen naar wens hiervan een struikelblok.
Door code aan te passen en verschillende parameters mee te geven aan het middelste gedeelte van de GAN is er getracht een vector te vinden die gezichten genereerde met een positieve of negatieve mond.
Hier is gebruik gemaakt van een .npy bestand “stylegan\_ffhq\_smile\_boundary” voor het aanpassen van de “z-space” of “stylegan\_ffhq\_smile\_w\_boundary” voor de “w-space” van het neurale netwerk.
Dit bestand is een “boundary vector” dat een richting definieert in de “latent space” van het StyleGAN model. Deze vector aanpassen zou glimlachen moeten aanpassen.
(Shen, 2024)
Door scripting die verschillende vectoren invoert in de GAN is de uitkomst van het model getracht de beïnvloeden naar onze wensen. De resultaten waren niet toereikend maar leverde interessante beelden op:
![A person wearing glasses and smiling
Description automatically generated](data:image/jpeg;base64...) werd -> ![A person with glasses smiling
Description automatically generated](data:image/jpeg;base64...)
Hieruit bleek dat deze vooraf bepaalde vector niet bruikbaar was en dat er een eigen vector gemaakt moest worden.
Door verschillende “seeds” te gebruiken met emoties die wij zochten is een Z vector berekend in een .npy vector formaat die later met het genereren van afbeeldingen gebruikt kan worden.
Een functie interpoleert tussen de gewichten voor het neurale netwerk en slaat deze in dit vector op. Onderstaand is een transitie tussen twee “seeds” te zien.
![A close-up of a person
Description automatically generated](data:image/png;base64...)![A close-up of a person
Description automatically generated](data:image/png;base64...) ![A close-up of a person smiling
Description automatically generated](data:image/png;base64...)
#### Conclusie
Door verder onderzoek te doen in fase twee naar het vinden van de juiste W en/of Z vector is het mogelijk StyleGAN2 te beïnvloeden om gewenste gezichten te genereren.
Hier hebben we een “seed” nodig voor alle gezichtsuitdrukkingen beschreven in de functionele vereisten en de code toegevoegd in bijlagen.
##### GAN websites
Het is ook mogelijk om ons model te trainen met willekeurig gegenereerde data van verschillende GAN-websites. Hoewel deze methode minder betrouwbaar is, kan het wel als optie worden overwogen.
Opvallend aan deze websites is dat zij vaak positieve emoties genereren.
### Voorbeelden hiervan zijn:

| <https://www.unrealperson.com/> | Genereert een afbeelding bij het drukken op de “generate” knop en is gratis te gebruiken |
| --- | --- |
| <https://thispersondoesnotexist.com/> | Genereert bij elke refresh een nieuw gezicht met behulp van GAN-technologie. Breed scala aan leeftijden en etniciteit. (Maakt gebruik van StyleGAN2). Parameters lijken ingesteld voor positieve foto’s.  Grote voordeel aan deze website is dat het met een http-verzoek een afbeelding toont zonder overhead een hierdoor met scripting eenvoudig is een groot aantal foto’s te vergaren. |
| <https://generated.photos/> | Meer dan 2 miljoen gegenereerde diverse gezichten. Alleen sorteer baar op neutraal en blij. Hierdoor is het niet mogelijk om te trainen op fronzen. Groot nadeel is dat het een betaalde service is als je meerdere foto’s wilt downloaden. |
#### Talk to edit
De Talk-to-Edit-repository is een tool voor gezichtsmanipulatie die bestaande afbeeldingen aanpast op basis van de parameters die de gebruiker opgeeft. De software stelt de gebruiker in staat via dialoog te communiceren met het model en geeft suggesties op basis van gebruikersinvoer.![A collage of a person's face
Description automatically generated](data:image/png;base64...)
Dit model is getraind met een aangepaste versie van de CelebA-dataset (Ziwei Lium, 2024) , met aangepaste labels die aansluiten bij de “gesprekken” die je met de software kunt voeren.
De software maakt gebruik van tekstverwerkingsmodellen en een taalencoder om tekstuele beschrijvingen af te stemmen op gezichtskenmerken in de “feauture space” en voorspellingen te doen.
Daarna bewerkt een GAN de afbeeldingen in hoge kwaliteit. (Yuming Jiang, 2021)
Hiermee kunnen hoge-kwaliteitsfoto's worden gebruikt zonder auteursrechten, bijvoorbeeld afkomstig van GAN-gedreven websites of bestaande datasets.
Met de juiste parameters, die via de dialogen worden bepaald, kunnen deze parameters worden toegepast in het script editing\_wo\_dialog.py, wat batchbewerkingen mogelijk maakt.
Dit maakt Talk-to-Edit een zeer interessante, bruikbare en automatisch inzetbare methode om afbeeldingen met diverse gezichtsuitdrukkingen te verkrijgen, zonder hoge hardware vereisten.
#### Conclusie
Helaas is deze methode niet toepasbaar door de dependencies in het “YAML” bestand voor de Conda packetmanager. Deze dependencies zijn te oud en verplaatst uit de hoofdrepository en hierdoor zal het installeren van deze specifieke package versies in ons tijdsbestek een te grote taak zijn en laten we dit onderzoek vallen.
### Zelfgemaakte – niet synthetische dataset
Het is mogelijk een eigen dataset te maken om het model te trainen dat specifiek voldoet aan de vereisten van het project. Het zelf maken van een dataset stelt ons in staat gericht data te verzamelen die hierop aansluiten.
#### Voordelen van een zelfgemaakte dataset
### 1. Gerichtheid op specifieke veresiten:

Met een eigen dataset draag je zorg voor het perfect aansluiten van de data op het te trainen model. Hierin heb je zelf de controle om onder dezelfde condities afbeeldingen te maken.
1. Controle over kwaliteit
Door zelf een dataset te maken heb je de kwaliteit van de dataset zelf in handen en ben je niet afhankelijk van de nauwkeurigheid van andere onderzoekers.
#### Uitdagingen in het maken van een eigen dataset
1. Bias
Door een eigen dataset te maken ontstaat een risico op bias. Dit komt voor doordat de dataset specifiek is afgestemd op een bepaald scenario, waardoor het getrainde model mogelijk niet om kan gaan met veranderingen en verschillen in personen. Het Google Fasemesh model vangt het probleem van verschillen in personen af door hun algoritme.
1. Arbeidsintensief
Het maken van een goede, en vooral grote, dataset is erg arbeidsintensief. Elke uitdrukking moet worden herhaald en worden gelabeld. Daarnaast is het voor diversiteit een uitdaging om een diverse groep bereid te krijgen om foto’s te laten maken voor het trainen van het model. In dit project zullen we focussen op gezichten van de onderzoekers.
3. Variatie in gezichtsuitdrukkingen
Een gezichtsuitdrukking is subjectief en per persoon verschillend. Door een te kleine steekproefomvang te nemen lopen we risico op een niet uniforme prestatie van het model.
### Conclusie
Uit voorafgaand onderzoek is de conclusie dat we twee richtingen op splitsen met het onderzoeksteam. Twee leden gaan een dataset maken met eigen gezichten, hierbij worden 20 foto’s voor elke uitdrukking gemaakt van elk project lid. Ook worden er meer gezichten toegevoegd als het onderzoeksteam vrijwilligers bereid krijgen.
Het tweede deel van het onderzoeksteam duikt verder in StyleGAN, om een geautomatiseerd proces te creëren.
# Deelvraag 3 – Computer vision
In Deelvraag 1 – Emotie classificatie is uitgelegd hoe mensen emoties kunnen uitdrukken in een glimlach, grote lach of frons. Een computer vision systeem moet de emoties op een andere analyseren dan hoe mensen dat doen. Dit hoofdstuk legt uit hoe een computer vision systeem een foto analyseert en onderscheid kan maken tussen glimlach, grote lach en fronsen.
## Overview
In het kort, werkt een computer vision systeem doorgaans met de volgende stappen:
1. Acquire
   Dit is het proces van het verkrijgen of verzamelen van de data die nodig is. Dit kan komen van bronnen zoals afbeeldingen, sensoren of andere datasets die relevant zijn voor het project.
2. Preprocessing
   Data wordt klaargemaakt voor analyse. Denk aan het opschonen, schalen en omzetten van data om het consistent en bruikbaar te maken voor verdere stappen.
3. Segmentation
   Het opdelen van data in kleinere, zinvolle delen voor nauwkeurige analyse
4. Labeling
   Het toewijzen van labels aan verschillende delen van de data voor herkenning.
5. Feature extraction
   Het selecteren van belangrijke eigenschappen in de data om patronen te identificeren.
6. Classification
   Het indelen van data in categorieën op basis van herkenbare patronen
Deze 6 stappen zijn de algemene stappen die een computer vision systeem doorloopt. (Géron, 2019)
## Feature extraction
Voor het analyseren van gezichtsuitdrukkingen in niet-bewegende plaatjes, bestaan 2 soorten feature extraction methodes:
1. Geometrie gebaseerde extractie
   Deze methode focust op het lokaliseren van specifieke punten op het gezicht, zoals de hoeken van de mond, ogen en wenkbrauwen. Aan de hand van de verhoudingen en posities tussen deze punten wordt een gezichtsuitdrukking herkend.
2. Uiterlijk gebaseerde extractie
   Deze methode analyseert texturen en schaduwen op het gezicht om kenmerken zoals rimpels en lijnen vast te leggen. Dit helpt om veranderingen in gezichtsuitdrukkingen te detecteren, zoals het optrekken van de wangen of het fronsen van de wenkbrauwen.
De vereisten van dit project beperkt dit rapport tot geometrie gebaseerde feature extractie. Google’s facemesh is namelijk een model gebaseerd op geometrie van een gezicht.
In computer vision systemen zijn bepaalde gezichtskenmerken nuttig om emoties te onderscheiden. Belangrijke kenmerken zijn:
1. Vorm en beweging van de mond
   De positie van de mond is een belangrijke aanwijzing voor emoties. Een opwaartse curve geeft blijheid aan, terwijl een neerwaartse curve verdriet suggereert. De mate van openheid kan ook verbazing of angst aangeven.
2. Positie en vorm van de wenkbrauwen
   Wenkbrauwen spelen een grote rol in het uiten van emoties. Opgetrokken wenkbrauwen kunnen verrassing betekenen, terwijl gefronste wenkbrauwen vaak boosheid of verwarring aangeven. Beweging en positie van de wenkbrauwen zijn cruciaal voor het correct herkennen van emoties.
3. Vorm van ogen
   Veranderingen in de vorm van de ogen, zoals verbreden of samenknijpen, geven emoties aan zoals angst of argwaan. Ook de richting van iemands blik kan context geven aan de gezichtsuitdrukking.
4. Beweging van de wangen
   Het optrekken van de wangen, vaak met rimpels bij de ogen, is een veelvoorkomend teken van lachen of blijheid.
5. Rimpels bij neus en voorhoofd
   Rimpels bij de neus kunnen walging aangeven, terwijl rimpels op het voorhoofd zorgen of diepe gedachten suggereren.
Computer vision systemen analyseren deze kenmerken om een zo accuraat mogelijke voorspelling te doen van het type gezichtsuitdrukking. (Huang, 2019)
Op het moment van schrijven is nog geen onderzoek gepubliceerd waarin het belang per feature gekwantificeerd wordt. Het onderzoek in dit rapport zal gebaseerd worden op geometrische analyse van wenkbrauwen en mond. Het computer vision systeem zal landmark features extraheren met Google Face Mesh. Landmark-features zijn specifieke punten op een gezicht die belangrijk zijn voor het herkennen van gezichtsuitdrukkingen en emoties. Deze punten kunnen bijvoorbeeld de hoeken van de ogen, de uiteinden van de lippen of de bovenkant van de neus zijn. Door de posities van deze punten te analyseren, kan een computer begrijpen hoe het gezicht beweegt en welke emotie of uitdrukking wordt getoond.
Met software zoals Google Face Mesh worden honderden van deze punten op het gezicht geplaatst, waardoor er een soort “kaart” ontstaat van het gezicht. Deze kaart helpt om veranderingen in uitdrukking nauwkeurig te volgen. Meer informatie over Google Face mesh is te vinden in Deelvraag 5 - Google facemesh.
# Deelvraag 4 – Algoritme voor usecase
Welk algoritme kunnen we het beste gebruiken met de vergaarde data?
## Support vector machine
Support vector machines (SVM) is een veelgebruikt machine-learning model wat lineaire, niet-lineaire classificatie, regressie en outlier-detectie kan uitveoren. Hierdoor is het een van de meest pouplaire machine-learning modellen.
Een SVM maakt een keuze aan de hand van lijnen,valt mijn data binnen of buiten mijn “support vector(s)”.
### Lineaire SVM
Een lineaire SVM kan een keuze maken tussen verschillende gelabelde klassen aan de hand van ingevoerde data. Het model trekt een lijn tussen training data en de klassen die hier aan toegewezen zijn. Dit model werkt goed als de berekende features duidelijk van elkaar te scheiden zijn met een rechte lijn: ![A diagram of a line graph
Description automatically generated with medium confidence](data:image/png;base64...)
(Géron, 2019)
### Niet lineaire SVM
Ook zijn SVM’s in te richten als niet lineair. Dit kan gedaan worden door een kernel uit te laten rekenen en dit net zo laten werken alsof je meerdere polynomen met de hand toevoegt.
### Dit is uit te voeren met de SVC klasse van sklearn:

![A computer code with text
Description automatically generated with medium confidence](data:image/png;base64...)
Deze code traint een SVM derdegraads polynoom om over een dataset heen te passen. Door parameters van de functie aan te passen kan de juiste SVM worden gekozen:
![A graph of a function
Description automatically generated with medium confidence](data:image/png;base64...)
Ook is het mogelijk een vector te maken met een Gaussich RBF kernel. In dit voorbeeld wordt een SVC-klasse gemaakt met een Gaussisch RBF kernel:
![A computer code with text
Description automatically generated with medium confidence](data:image/png;base64...)
Ook hier zijn in te stellen parameters mee te geven om de juiste vector te bepalen. Waarin de gamma de belangrijkste is;
Een te hoge gamma maakt de kromme smaller waardoor het model gevoelig wordt voor individuele datapunten. Een te lage gamme zorgt voor een bredere kromme en daardoor beslissingsgrens. Dit kan erg nuttig zijn bij overfitting.
![A screenshot of a graph
Description automatically generated](data:image/png;base64...)
(Géron, 2019)
## K-Means clustering
K-means clustering is een algoritme dat een dataset verdeelt in een vooraf bepaald aantal clusters. Dit aantal clusters wordt aangeduid met de variabele k. Het algoritme werkt iteratief om de data in k clusters op te splitsen. In elk cluster worden data-punten gegroepeerd die qua eigenschappen dicht bij elkaar liggen. K-means bepaalt eerst willekeurig k middelpuntpunten en verdeelt vervolgens de data rondom deze middelpunten. Na elke iteratie worden de middelpunten herberekend om zo een optimale verdeling te bereiken. Het proces stopt wanneer de middelpunten niet meer veranderen of wanneer het maximale aantal iteraties is bereikt.
### Belangrijke eigenschappen die k-NN populair maken:

* **Gebruikt data direct**k-NN creëert geen model of probeert niet vooraf een patroon in de data te begrijpen. In plaats daarvan onthoudt het gewoon alle voorbeelden die het heeft gezien, en wanneer een nieuw punt binnenkomt, kijkt het naar de dichtstbijzijnde voorbeelden om een beslissing te nemen. Dit maakt k-NN eenvoudig om in te stellen, maar het kan traag zijn met veel data.
* **Makkelijk te tunen**
  k-NN is heel eenvoudig. Je hoeft alleen maar het aantal buren (k) in te stellen, en het gebruikt de dichtstbijzijnde punten om te beslissen over de categorie van het nieuwe punt. Dit maakt het een makkelijk model voor classificatie voor beginners.
* **De juiste hyperparameter**
  Het aantal buren, of “k,” kan de nauwkeurigheid beïnvloeden. Als k te klein is, kan het algoritme verward raken door ruis (willekeurige punten). Als k te groot is, kan het details missen. Het testen van verschillende waarden voor k kan helpen om de beste resultaten te behalen.
* **Geen aannames over de data**
  k-NN maakt geen aannames over de vorm of verspreiding van de data, wat het flexibel maakt. Maar het kan traag zijn als er veel punten zijn om te controleren, omdat het elk nieuw punt vergelijkt met elk voorbeeld.
* **Gevoelig voor afstand**Omdat k-NN kijkt naar afstanden tussen punten, is het essentieel dat alle kenmerken (zoals lengte en gewicht) op dezelfde schaal staan. Anders kan één kenmerk de afstandsbepaling domineren, wat tot slechte resultaten kan leiden.
* **Werkt het best met kleine datasets**
  k-NN is het meest effectief met kleine datasets, omdat het elke keer elk voorbeeld moet opslaan en vergelijken. Bij grote datasets kan het langzaam en minder praktisch worden.
Kort samengevat is k-Nearest Neighbors (k-NN) een eenvoudige en handige methode om data in groepen te plaatsen op basis van gelijkenis. Voor het herkennen van gezichtsuitdrukkingen zou k-NN goed kunnen werken, zolang de dataset klein is en de data goed is voorbereid. Omdat er echter veel verschillen kunnen zijn in de afstanden tussen de datapunten, kan k-NN soms minder nauwkeurig zijn. Als de kenmerken van gezichtsuitdrukkingen verschillende schalen hebben, kan één kenmerk de afstandsmeting overheersen, waardoor het model fouten maakt in de classificatie.
Om k-NN goed te laten werken, is het belangrijk om de data qua schaal gelijk te maken (bijvoorbeeld door normalisatie of standaardisatie), zodat alle kenmerken eerlijk meetellen. Zonder deze stap zal het model niet nauwkeurig zijn. (IBM)
## Decision tree
Een decision tree is een populair machine learning-model dat gebruikt wordt voor zowel classificatie- als regressietaken. Het werkt als een stroomdiagram, waarbij elke beslissing leidt tot verdere vertakkingen, waarbij de gegevens stap voor stap worden opgesplitst totdat er een eindbeslissing of "blad" is bereikt. Bijvoorbeeld, in een decision tree voor het classificeren van dieren kan de eerste vraag zijn: "Heeft het veren?" Dit leidt tot verschillende vertakkingen afhankelijk van het antwoord. Door deze structuur zijn decision trees eenvoudig te begrijpen en te interpreteren.
### Belangrijkste kenmerken van een decision tree:

1. **Eenvoudig te begrijpen**
   Beslissingsbomen zijn gemakkelijk te volgen omdat ze gebruik maken van een eenvoudig, visueel stroomdiagram met duidelijke beslissingen. Deze structuur maakt het gemakkelijk om te zien welke kenmerken of gegevenspunten het belangrijkst zijn, in tegenstelling tot meer complexe modellen zoals neurale netwerken.
2. **Minimale gegevensvoorbereiding**
   Beslissingsbomen hebben niet veel gegevensschoonmaak of -opmaak nodig. Ze werken goed met verschillende soorten data, of het nu gaat om getallen of categorieën. Ze kunnen ook omgaan met ontbrekende waarden, wat nuttig is aangezien veel modellen moeite hebben met hiaten in de gegevens.
3. **Flexibel voor verschillende taken**
   Beslissingsbomen kunnen worden gebruikt voor zowel classificatie (het sorteren van gegevens in groepen) als regressie (het voorspellen van een getal), wat ze zeer veelzijdig maakt. Ze worden niet beïnvloed door relaties tussen kenmerken, dus als twee kenmerken nauw met elkaar verbonden zijn, zal de boom er gewoon één kiezen om op te splitsen.
4. **Kan makkelijk overfitten**Beslissingsbomen kunnen te complex worden en mogelijk alleen goed werken met de trainingsgegevens, maar niet met nieuwe gegevens. Om dit te voorkomen kunnen ze worden "gesnoeid" (onnodige vertakkingen worden verwijderd), tijdens of na het bouwen van de boom.
5. **Gevoelig voor kleine veranderingen**
   Een kleine verandering in de gegevens kan leiden tot een heel andere beslissingsboom. Dit betekent dat ze sterk kunnen variëren, wat "hoge variantie" wordt genoemd. Het gebruik van technieken zoals "bagging" (het middelen van voorspellingen over veel bomen) kan helpen om dit te verminderen, maar kan nog steeds resulteren in voorspellingen die erg veel op elkaar lijken.
6. **Duurder om te trainen**
   Het bouwen van een decision tree kan trager en intensiever zijn dan andere algoritmen en kost meer rekenkracht om te trainen dan andere modellen.
Een decision tree zou kunnen werken voor het classificeren van gezichtsuitdrukkingen als de dataset eenvoudig en klein is. Beslissingsbomen zijn gemakkelijk te interpreteren en kunnen snelle beslissingen nemen op basis van specifieke kenmerken, wat nuttig is voor eenvoudige classificatietaken. Echter, als de afbeeldingen complexe patronen bevatten of als er veel ruis is (irrelevante kenmerken), kunnen beslissingsbomen moeite hebben met nauwkeurigheid of overfitting. (IBM, sd) (Kumar, sd)
## Lineaire regressie
Lineaire regressie Is een machine learning model dat een waarde van een variabele voorspeld afhankelijk van een waarde van een andere variabele. De regressie is eigenlijk het fitten van een lineare vergelijking door middel van het aanpassen van de coëfficiënten in de vergelijking. Net zo lang tot dat er een optimum gevonden is wat inhoudt dat de opgetelde som van de fout ten opzichte van de lineare vergelijking het kleinst is. Nadat de lijn is passend gemaakt kan een waarde bijvoorbeeld y worden voorspeld voor een bepaalde x.
### ![](data:image/jpeg;base64...)Kenmerken van een lineair regressie model:

* Relatief simpel met een makkelijk te begrijpen wiskundige formule
* Neemt aan dat de 2 variabelen in de vergelijking een linear verband hebben.
* Heeft minder data nodig om een succesvol model te creëren in vergelijking met andere meer complexere algoritmes
* Is gevoeliger voor uitschieters (outliers).
De eenvoud van lineaire regressie maakt het een makkelijk model om mee te werken, maar of dit model geschikt is voor het analyseren van gezichtsuitdrukkingen, valt te betwijfelen. (GeeksForGeeks, 2024)
## Autoencoders
Een auto encoder is een type unsupervised machine learning neuraal netwerk. Deze is gedesigned om efficiënt belangrijke features van input data te extraheren en daarna wordt de originele data gereconstrueerd.
Auto encoders gebruiken een algoritme dat probeert om variabelen te vinden wat bepaalde data kan onderscheiden. Soms zijn dit variabelen die op een eerste blik niet zichtbaar zijn. Ook kijkt het welke variabelen het meest succesvol zijn in het reconstrueren van de originele data.
Twee vormen van auto encoders met name variational autoencoders (VAEs) en adversarial autoencoders (AAEs) zijn aangepast voor gebruik in generatieve applicaties zoals foto generatie of andere data.
Verder worden autoencoders voor een breed scala aan toepassingen gebruikt. Denk aan bijv denoising, abnormaliteits detectie, generatie.
### Eigenschappen van autoencoders zijn als volgt:

1. Efficiënte data representatie, de grootte van de data word verkleint zonder verlies van essentiële informatie.
2. Heeft geen gelabelde data nodig, wat een dataset opstellen een stuk minder intensief maakt
3. Erg hoge computatie kosten in vergelijking met niet neural netwerk algoritmes
4. Risico voor over fitting waarbij de autoencoder data onthoudt in plaats van features
5. Heeft meer fine-tuning nodig in vergelijking met andere algoritmes
## Conclusie
Zodra de dataset klaar is, zullen we verschillende algoritmes testen om te bepalen welk model het beste werkt voor het herkennen van emoties met face mesh data. Op basis van de theorie die eerder in dit document is beschreven, lijkt een Support Vector Machine (SVM) een aantrekkelijke keuze.
SVM is goed in het maken van duidelijk onderscheid tussen groepen, zoals verschillende gezichtsuitdrukkingen, en is daarnaast relatief eenvoudig te trainen met minder data dan sommige andere modellen. In de volgende fase zullen we onderzoeken of SVM of een ander model uiteindelijk het meest geschikt is voor onze toepassing.
# Deelvraag 5 - Google facemesh
Google Face mesh detection is een real-time “gaas” model met hoge betrouwbaarheid dat 468 3D-punten tekent over een menselijk gezicht.
### Terminologie van Google vertaald naar Nederlands:

| Bounding box | Rechthoekige omtrek voor een gedetecteerd gezicht | A person with a grid on her face |
| --- | --- | --- |
| Facemesh info | Een groep van 468 3D punten en lijnen dat gebruikt kan worden om een geometrisch gaas te tekenen over een gedetecteerd gezicht. |
De API is bedoeld om nauwkeurige operaties te doen op gezichten in real time zoals:
1. AR-Filters
2. Selfie-opnames
3. Real time gezichtsprojectie op een 3D-model
Voor ons project kunnen wij de gedetecteerde facemesh gegevens gebruiken en filteren om delen van het gezicht te isoleren en te laten classificeren door ons het machine learning model. (Google, 2024)
Het 3D gaasmodel is uit te zetten in 2D en laat zien dat het gaas bestaat uit 468-3D punten en driehoek-informatie.
Elke driehoek heeft 3 3-D punten. Bijvoorbeeld #0,#37 en #164 wat een klein driehoekige oppervlakte representeert tussen de neus en de lippen. (Google, 2024)
![Example of face mesh info, click for zoomable image](data:image/png;base64...)
## Mediapipe Face mesh
Mediapipe is een open-source framework van Google, waar Face Mesh een onderdeel van uitmaakt. Het maakt gebruik van machine learning om zonder 3D interface met een enkele 2-D camera een 3D model te maken van iemands gezicht. Door gebruik te maken van een lichtgewicht model, GPU-acceleratie door de pipeline stelt het de gebruiker in staat om real-time performance te verkrijgen. (Google, 2024)
### Face Mesh pipeline
Het hele beeld van de camera wordt verwerkt door een lichtgewicht gezichtsdetector die een rechthoek om het gezicht te tekenen als zowel een aantal kenmerken (b.v. ogen, oren en neus). Deze kenmerken worden vervolgens gebruikt om het gezicht uit te lijnen aan de hand het midden van de ogen en de horizontale as van de rechthoek.
Hierna wordt het gezicht uit het beeld uitgeknipt, hervormt naar een 128x128 tot 256x256 afbeelding en gebruikt als input voor het “mesh prediction neural network”. Het gaas wordt gegenereerd in 2D en vervolgens worden de Z-coördinaten geïnterpreteerd als diepte ten opzichte van het referentievlak dat door het zwaartepunt van het mesh gaat. Dit wordt geschaald zodat er een vast beeldverhouding blijft ongeacht de afstand van de camera. (Grundmann, 2019)
Dit framework stelt ons in staat om een vast grid rooster te krijgen over het gezicht van een mens en deze vaste datapunten te gebruiken in ons machine learning model.
### Gebruik Mediapipe Face Mesh
MediaPipe is te installeren als klaar voor gebruik aanpasbaar Python package.
De voorbeeldcode van Google is bruikbaar met de volgende Python Pip dependencies: mediapipe, numpy en matplotlib.
Op de voorbeeldcode in bijlage “Voorbeeldcode Google/Mediapipe facemesh” kunnen we door ontwikkelen. (Google, 2024)
## Gelaatsdelen herkennen uit het gaas
Uit het gaas zijn de volgende gelaatsdelen te definiëren en te importeren direct uit mediapipe en is gedefinieerd in “mediapipe/python/solutions/face\_mesh.py”
In dit python bestand wordt “face\_mesh\_connections” geïmporteerd en zijn de onderdelen van het “Face Mesh” gaas waar we direct gebruik van kunnen maken als input voor ons model.
![A screen shot of a computer screen
Description automatically generated](data:image/png;base64...)
Een voorbeeld waar wij veelvoudig gebruik van kunnen maken is het onderdeel FACEMESH\_LIPS. Hier zijn de volgende punten op het “Face Mesh” gaas voor-gedefinieerd:
![A screenshot of a computer screen
Description automatically generated](data:image/png;base64...)
Aan de hand van de afstanden van deze coördinaten kunnen we rekenen en ons machine-learning model mee gaan trainen.
## Conclusie
Face mesh is voor ons project uitstekend toe te passen door de volgende voor gedefinieerde onderdelen uit het gaas te gebruiken:
![A computer screen shot of a computer screen
Description automatically generated](data:image/png;base64...)
Door een script automatisch over onze vooraf gelabelde training data te laten gaan, de wenkbrauwen en monden extraheren en de featuredata hiervan bepalen. We hebben ervoor gekozen om geen ogen te gebruiken in onze analyse, omdat dit onnodige verwarring kan veroorzaken en niet nodig is voor het detecteren van de gezichtsuitdrukkingen die wij willen onderscheiden.
# Deelvraag 6 – Ethische overwegingen
**Welke ethische overwegingen zijn genomen bij gezichtsuitdrukking detectie?**
Bij het gebruik van technologie zoals Google Face Mesh om gezichtsuitdrukkingen te herkennen, zijn er een paar belangrijke ethische punten om over na te denken. Ten eerste is privacy en toestemming erg belangrijk. Mensen moeten duidelijk toestemming geven voordat hun gezichtsdata wordt verzameld en gebruikt. Zonder toestemming kan dit de privacy van mensen schenden.
**Privacy en Gegevensbescherming**
Een van de belangrijkste ethische overwegingen bij het gebruik van gezichtsuitdrukking detectie is privacy. Omdat deze technologie gezichtsdata verzamelt en analyseert, moet er zorgvuldig worden omgegaan met de bescherming van persoonlijke informatie. Het is belangrijk dat bedrijven die met deze technologie werken duidelijke en strikte privacy beleidsregels hanteren om te voorkomen dat persoonlijke gegevens misbruikt of onveilig opgeslagen worden. Een mogelijke oplossing hiervoor is het implementeren van technieken waarbij de verwerking van gegevens lokaal op het apparaat van de gebruiker plaatsvindt, zodat de data niet wordt gedeeld met externe servers. Dit minimaliseert de kans op datalekken en waarborgt de privacy van de gebruiker. Ook is het belangrijk hoe deze gegevens worden bewaard en beschermd. Gezichtsdata is gevoelige informatie en moet veilig worden opgeslagen, bijvoorbeeld door versleuteling en door te voldoen aan wetten zoals de AVG. Zo voorkom je dat onbevoegden toegang krijgen tot de gegevens.
**Transparantie en Misbruik**
Naast privacy en bias is transparantie over het gebruik van gezichtsuitdrukking detectie ook van groot belang. Gebruikers moeten weten wanneer en hoe hun gezichtsdata wordt gebruikt. Zonder deze transparantie kan de technologie worden ingezet voor verborgen monitoring of zelfs manipulatie, wat ethische vragen oproept over toestemming en autonomie. Om misbruik te voorkomen, moeten bedrijven duidelijke richtlijnen hebben voor hoe de technologie mag worden gebruikt en moeten er toezichtmechanismen zijn om ervoor te zorgen dat deze richtlijnen worden nageleefd. Het is dus belangrijk om een goede balans te vinden tussen de voordelen van de technologie en de verantwoordelijkheid die ermee komt (Bargagni, 2023).
**Opslaan van landmarks**
Het opslaan van gezichtslandmarks lijkt op het eerste gezicht misschien onschuldig omdat het alleen om numerieke gegevens gaat (zoals coördinaten die specifieke punten op het gezicht aanduiden). Toch roept het ethische vragen op.
Landmarks zijn afgeleid van een gezicht, wat een unieke persoonlijke eigenschap is. Hoewel de gegevens zelf cijfermatig zijn, kunnen ze gecombineerd worden met andere informatie om mensen te identificeren, zeker als ze herleidbaar zijn tot een specifiek gezicht. Als ze in een dataset worden opgeslagen zonder anonimisatie, kunnen ze bij grote datasets of datasets met bekende gezichten gebruikt worden om personen alsnog te identificeren. Dit kan een risico vormen voor de privacy van een persoon, vooral als de gegevens zonder hun toestemming worden gedeeld of verkocht (Hassan, 9).
**Conclusie**
Voor dit project hebben we ervoor gekozen om geen persoonsdata te exporteren of op te slaan. Dit komt doordat het project gericht is op het ontwikkelen van een API voor een robot, waarbij alleen gezichtsuitdrukkingen en tijdstippen worden geregistreerd. Dit helpt om de privacy van gebruikers te waarborgen en ervoor te zorgen dat gevoelige gegevens beschermd blijven. Door deze aanpak minimaliseren we het risico van gegevensmisbruik en voldoen we aan de belangrijke ethische richtlijnen rondom privacy en transparantie.
# Bronnen
Bargagni, S. (2023, Januari 25). Opgehaald van Morphcast: https://www.morphcast.com/blog/ethic-and-responsible-use-of-face-emotion-recognition/
Chauhan, N. S. (2024, 11 2). *Generate Realistic Human Face using GAN*. Opgehaald van kaggle.com: https://www.kaggle.com/code/nageshsingh/generate-realistic-human-face-using-gan
DDS, R. S. (2005, 03). *The Eight Components of a Balanced Smile*. Opgehaald van Journal of clinical orthodontics: https://www.jco-online.com/archive/2005/03/155-overview-the-eight-components-of-a-balanced-smile/
*Facial expression*. (2018, 04 19). Opgehaald van apa.org: https://dictionary.apa.org/facial-expression
GeeksForGeeks. (2024, 10 23). *Linear Regression in Machine learning*. Opgehaald van geeksforgeeks.org: https://www.geeksforgeeks.org/ml-linear-regression/
Géron, A. (2019). *Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow.* O'REILLY.
Google. (2024, 11 5). *developers.google.com*. Opgehaald van Face mesh detection: https://developers.google.com/ml-kit/vision/face-mesh-detection
Google. (2024, 11 5). *Face mesh detection concepts*. Opgehaald van developers.google.com: https://developers.google.com/ml-kit/vision/face-mesh-detection/concepts
Google. (2024, 5 11). *MediaPipe Face Mesh*. Opgehaald van github.com: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face\_mesh.md
Google. (2024, 5 11). *python.md*. Opgehaald van github: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting\_started/python.md
Grundmann, Y. K. (2019). *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs.* Mountain View: Google. Opgehaald van https://arxiv.org/abs/1907.06724
Hassan. (9, July 2024). Opgehaald van Recognito: https://recognito.vision/facial-recognition-ethics-considerations/
Huang, Y. (2019, 09 20). *Facial Expression Recognition: A Survey.* Opgehaald van mdpi.com: https://www.mdpi.com/2073-8994/11/10/1189
IBM. (sd). *What is a decision tree?* Opgehaald van ibm.com: https://www.ibm.com/topics/decision-trees
IBM. (sd). *What is the k-nearest neighbors (KNN) algorithm?* Opgehaald van ibm.com: https://www.ibm.com/topics/knn
Jaffe, E. (2011, 02 11). *The Psychological Study of Smiling*. Opgehaald van psychologicalscience.org: https://www.psychologicalscience.org/observer/the-psychological-study-of-smiling
Kumar, V. (sd). *Decision Tree Algorithm overview explained*. Opgehaald van towardsmachinelearning.org: https://towardsmachinelearning.org/decision-tree-algorithm/
NVIDIA. (2024, 11 2). *NVlabs stylegan2*. Opgehaald van github.com: https://github.com/NVlabs/stylegan2
Sendic, G. (2023, 11 21). *Facial muscles*. Opgehaald van kenhub.com: https://www.kenhub.com/en/library/anatomy/the-facial-muscles
Yuming Jiang, Z. H. (2021). *Talk-to-Edit: Fine-Grained Facial Editing via Dialog.* Hong Kong: IEEE International Conference on Computer Vision (ICCV). Opgehaald van https://arxiv.org/abs/2109.04425
Ziwei Lium, P. L. (2024, 11 2). *Large-scale CelebFaces Attributes (CelebA) Dataset*. Opgehaald van mmlab.ie.cuhk.edu.hk: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# Bijlagen
## Voorbeeldcode Google/Mediapipe facemesh
### Face\_landmark\_detection.py
![A screen shot of a computer program
Description automatically generated](data:image/png;base64...)
### Face\_landmark\_visualization.py
![A screen shot of a computer program
Description automatically generated](data:image/png;base64...)
![A screen shot of a computer program
Description automatically generated](data:image/png;base64...)
### run\_facemark\_pipeline.py
![A screenshot of a computer program
Description automatically generated](data:image/png;base64...)
### StyleGAN modificatie
#### Geinterpoleerde afbeeldingen generen
### Originele functie Nvidia met extra commentaar:

![A screenshot of a computer program
Description automatically generated](data:image/png;base64...)
Gemodificeerde versie voor genereren van Z vectoren
![A screenshot of a computer program
Description automatically generated](data:image/png;base64...)
### Geneste functie in bovenstaand om afbeeldingen en vectoren op te slaan:

![A screen shot of a computer program
Description automatically generated](data:image/png;base64...)
#### Interpolatie vecotoren
### Interpolatie tussen twee latente vectoren van de GAN input seeds:

![A screen shot of a computer program
Description automatically generated](data:image/png;base64...)
