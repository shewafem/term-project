from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

database = {
    "insulin": "A peptide hormone produced by the beta cells of the pancreas, essential for regulating blood glucose "
               "levels. Insulin facilitates the uptake of glucose by cells for energy production or storage as glycogen, and it plays a critical role in lipid and protein metabolism.",
    "blood glucose": "The amount of glucose present in the bloodstream, typically measured in milligrams per "
                     "deciliter (mg/dL). Normal fasting blood glucose levels range from 70 to 99 mg/dL, and glucose serves as a primary energy source for cellular processes, especially in the brain and muscles.",
    "hyperglycemia": "A condition where blood glucose levels are higher than normal, typically above 125 mg/dL when "
                     "fasting. Chronic hyperglycemia is a hallmark of diabetes and can lead to complications such as neuropathy, retinopathy, and cardiovascular diseases.",
    "hyperglycemic clamps": "A precise experimental technique used in metabolic research to assess beta-cell function and insulin secretion. It involves infusing glucose intravenously to maintain a stable elevated blood glucose concentration, while measuring the amount of insulin the pancreas releases in response.",
    "metabolic syndrome": "A combination of metabolic disorders, including central obesity, insulin resistance, "
                          "dyslipidemia (elevated triglycerides and reduced HDL cholesterol), high blood pressure, and elevated fasting blood glucose. This syndrome significantly increases the risk of developing type 2 diabetes, cardiovascular disease, and stroke.",
    "candidemia": "A severe bloodstream infection caused by Candida species, which are opportunistic pathogens. "
                  "Candidemia can result from invasive medical procedures, the use of central venous catheters, or weakened immune systems. Symptoms may include fever, chills, and sepsis, requiring prompt antifungal treatment.",
    "yeasts": "Single-celled fungi that reproduce by budding or fission. While many yeasts, such as Saccharomyces "
              "cerevisiae, are harmless or beneficial, some, like Candida albicans, can become pathogenic under certain conditions, leading to infections in the skin, mucous membranes, or bloodstream.",
    "candida esophagitis": "An opportunistic fungal infection of the esophagus caused primarily by Candida albicans. "
                           "Common in immunocompromised individuals, it is characterized by painful swallowing (odynophagia), white plaques in the esophagus, and inflammation, often requiring antifungal therapy.",
    "endoscopy": "A medical procedure involving the insertion of a flexible tube equipped with a camera and light ("
                 "endoscope) into the body to visualize internal structures. Endoscopy is commonly used to diagnose gastrointestinal conditions, such as ulcers, esophagitis, or tumors, and may include biopsy collection.",
    "fungal infection": "A condition caused by the overgrowth or invasion of fungi into tissues, ranging from "
                        "superficial infections like athlete's foot and oral thrush to systemic and potentially life-threatening infections such as invasive aspergillosis or candidemia. These infections often require antifungal medications and may affect immunocompromised patients more severely."
}

model = Word2Vec.load("models/model_med.model")

def vector_num(query):
    query = query.lower()
    words = query.split()
    # print(words)
    word_vectors = []
    average_vector = []

    for word in words:
        if model.wv.has_index_for(word):
            word_vectors.append(model.wv.get_vector(word))
    
    # print(word_vectors)
    if len(word_vectors) > 0:
        average_vector = np.mean(word_vectors, axis=0)
    return average_vector


def directory_query(query):
    average_vector = vector_num(query)

    closest_key = None
    closest_similar = -1
    similar = -1

    for key, description in database.items():
        key_vector = vector_num(key)
        if (len(key_vector) > 0) & (len(average_vector) > 0):    
            similar = cosine_similarity([average_vector],[key_vector])[0][0]
        if similar > closest_similar:
            closest_similar = similar
            closest_key = key
        
        description_vector = vector_num(description)
        if (len(description_vector) > 0) & (len(average_vector) > 0):    
            similar = cosine_similarity([average_vector],[description_vector])[0][0]
        if similar > closest_similar:
            closest_similar = similar
            closest_key = key
    
    if closest_key is not None:
        print("Описание:", database[closest_key])
    else:
        print("Описание не найдено.")

user_query = input("Введите ваш запрос: ")
directory_query(user_query)