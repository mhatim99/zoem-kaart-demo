# 🐝 Zoem Kaart Web Demo

**Dutch Pollinator Biodiversity Explorer**

A web-based demonstration of the Zoem Kaart QGIS plugin, showcasing Dutch pollinator biodiversity analysis capabilities.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌐 Live Demo

**[Try the live demo →](https://zoem-kaart.streamlit.app)**  
*(Deploy your own instance - see instructions below)*

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🐝 **Multi-Taxon Support** | Explore wild bees, butterflies, and hoverflies |
| 🗺️ **Interactive Map** | Clustered markers with density heatmap |
| 📊 **Diversity Metrics** | Shannon, Simpson, and Pielou's evenness indices |
| 🔴 **Red List Integration** | Automatic Dutch conservation status enrichment |
| 📅 **Temporal Analysis** | Observation trends over time |
| 📥 **Data Export** | Download results as CSV |

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/zoem-kaart-demo.git
cd zoem-kaart-demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your forked repo
4. Set `app.py` as the main file
5. Click "Deploy"

## 📁 Project Structure

```
zoem-kaart-demo/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/
    └── config.toml       # Streamlit theme configuration
```

## 🔗 Data Sources

- **[GBIF](https://www.gbif.org/)** - Global Biodiversity Information Facility
- **Dutch Red List** - Conservation status classifications

## 📊 Full QGIS Plugin

This web demo showcases a subset of the full **Zoem Kaart QGIS Plugin** capabilities:

| Feature | Web Demo | QGIS Plugin |
|---------|:--------:|:-----------:|
| GBIF data | ✅ | ✅ |
| iNaturalist | ❌ | ✅ |
| Waarneming.nl | ❌ | ✅ |
| FLORON | ❌ | ✅ |
| Hotspot Analysis | ❌ | ✅ |
| Foraging Buffers | ❌ | ✅ |
| Beta Diversity | ❌ | ✅ |
| Species-Area Curves | ❌ | ✅ |
| Nectar/Pollen Index | ❌ | ✅ |
| Species Traits DB | ❌ | ✅ |

## 👤 Author

**Mohamed Z. Hatim, PhD**  
Vegetation and Landscape Ecology  
Wageningen University & Research  
📧 mohamed.hatim@wur.nl

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- GBIF for open biodiversity data
- Streamlit team for the amazing framework
- Dutch conservation organizations for Red List data
