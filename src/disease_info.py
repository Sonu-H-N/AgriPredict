"""
disease_info.py
----------------
Reference info for common rice leaf diseases: short description + basic
management tips. Used by the web app to give farmers actionable guidance
alongside the raw prediction, not just a label.

NOTE: These are general, educational summaries only — always advise users
to confirm with a local agricultural extension officer before applying any
chemical treatment.
"""

DISEASE_INFO = {
    "Bacterial_Leaf_Blight": {
        "description": (
            "A bacterial disease causing yellow-to-white stripes or lesions "
            "along the leaf veins, usually starting at the leaf tip or edges "
            "and spreading downward."
        ),
        "management": (
            "Use resistant rice varieties, avoid excess nitrogen fertilizer, "
            "ensure proper field drainage, and remove/destroy infected plant "
            "debris. Copper-based bactericides may help in early stages — "
            "consult a local agricultural officer for approved products."
        ),
        "severity": "High",
    },
    "Rice_Blast": {
        "description": (
            "A fungal disease producing diamond or spindle-shaped gray-white "
            "lesions with brown borders on leaves, and can also affect stems "
            "and grain necks. Considered the most destructive rice disease "
            "worldwide."
        ),
        "management": (
            "Use blast-resistant varieties, avoid excessive nitrogen, maintain "
            "balanced field water levels, and apply recommended fungicides "
            "(e.g., tricyclazole) at early signs if outbreak risk is high."
        ),
        "severity": "Very High",
    },
    "Brown_Spot": {
        "description": (
            "A fungal disease causing small circular brown-to-dark lesions "
            "scattered across the leaf surface. Often linked to poor soil "
            "fertility, especially potassium deficiency."
        ),
        "management": (
            "Improve soil fertility (balanced NPK, especially potassium), "
            "use certified disease-free seeds, and avoid water stress during "
            "the growing season."
        ),
        "severity": "Moderate",
    },
    "Leaf_Smut": {
        "description": (
            "A fungal disease producing small black angular spots (fungal "
            "fruiting bodies) scattered on the leaf surface, generally less "
            "damaging than blast or blight."
        ),
        "management": (
            "Generally minor — good field sanitation and balanced fertilization "
            "usually keep this under control. Fungicide treatment is rarely "
            "needed unless the outbreak is severe."
        ),
        "severity": "Low",
    },
    "Tungro": {
        "description": (
            "A viral disease spread by green leafhoppers, causing yellow-to-"
            "orange leaf discoloration and stunted plant growth. Can spread "
            "rapidly across a paddy field."
        ),
        "management": (
            "Control leafhopper populations (the disease vector) using "
            "recommended insecticides, use tungro-resistant varieties, and "
            "remove infected plants early to limit spread."
        ),
        "severity": "Very High",
    },
    "Healthy": {
        "description": "No visible signs of disease — leaf appears uniformly healthy.",
        "management": "Continue regular field monitoring and good agronomic practices.",
        "severity": "None",
    },
}


def get_disease_info(label):
    """
    Looks up disease info for a predicted class label. Falls back to a
    generic message if the label isn't in the reference dictionary (e.g.
    a custom class the user added).
    """
    return DISEASE_INFO.get(
        label,
        {
            "description": "No reference information available for this class yet.",
            "management": "Consult a local agricultural extension officer for guidance.",
            "severity": "Unknown",
        },
    )
