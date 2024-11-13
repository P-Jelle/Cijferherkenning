# ✍️ Cijfer Herkenning

Een eenvoudige webgebaseerde applicatie om cijfers (0-9) te tekenen en te herkennen met behulp van TensorFlow.js. Gebruikers kunnen hun getekende cijfers labelen en opslaan als trainingsdata om een neuraal netwerk te trainen. Vervolgens kan de applicatie cijfers voorspellen op basis van de getekende input.

---

## 📑 Inhoud

- [⚙️ Installatie en Setup](#-installatie-en-setup)
- [🔧 Gebruik](#-gebruik)
- [📂 Projectstructuur](#-projectstructuur)
- [✨ Functies](#-functies)
- [💻 Gebruikte Technologieën](#-gebruikte-technologieën)

---

## ⚙️ Installatie en Setup

1. **Repository Downloaden of Kloonen**: Download of kloon deze repository.
2. **Applicatie Starten**: Open `index.html` in een webbrowser.

## 🔧 Gebruik

1. **Start de Applicatie**: Open `index.html` in een webbrowser.
2. **Cijfer Tekenen**: Gebruik het 10x10 grid om een cijfer (0-9) te tekenen.
3. **Cijfer Labelen en Opslaan**:
   - Voer het getekende cijfer in het invoerveld in.
   - Klik op **Sla cijfer op** om het cijfer te labelen en op te slaan.
4. **Dataset Opbouwen**: Herhaal de bovenstaande stappen om meerdere cijfers te verzamelen.
5. **Voorspelling Uitvoeren**: Klik op **Voorspel** om een voorspelling te krijgen op basis van het getrainde model.

## 📂 Projectstructuur

- **`index.html`**: De hoofd-HTML-pagina van de applicatie.
- **`style.css`**: Bevat de stijlen voor de applicatie.
- **`script.js`**: Bevat de logica voor het tekengrid, gegevensopslag, modeltraining en voorspelfuncties.

## ✨ Functies

- **Interactie op het Grid**: Tekenen op een 10x10 grid.
- **Opslaan en Labelen**: Cijfers labelen en opslaan als trainingsdata.
- **Real-time Modeltraining**: Train het neuraal netwerk in de browser met TensorFlow.js.
- **Voorspelling**: Voorspel het getekende cijfer op basis van het getrainde model.
- **Grid Leegmaken**: Maakt het grid leeg voor een nieuwe tekening.

## 💻 Gebruikte Technologieën

- **HTML/CSS**: Structuur en styling van de applicatie.
- **JavaScript**: Voor de functionaliteit en interactie van de applicatie.
- **TensorFlow.js**: Voor het bouwen en trainen van een neuraal netwerk in de browser.

---
