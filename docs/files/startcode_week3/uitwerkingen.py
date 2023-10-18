import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# OPGAVE 1a
def plot_image(img, label):
    # Deze methode krijgt een matrix mee (in img) en een label dat correspondeert met het 
    # plaatje dat in de matrix is weergegeven. Zorg ervoor dat dit grafisch wordt weergegeven.
    # Maak gebruik van plt.cm.binary voor de cmap-parameter van plt.imgshow.

    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(label)
    plt.show()
    

# OPGAVE 1b
def scale_data(X):
    # Deze methode krijgt een matrix mee waarin getallen zijn opgeslagen van 0..m, en hij 
    # moet dezelfde matrix retourneren met waarden van 0..1. Deze methode moet werken voor 
    # alle maximale waarde die in de matrix voorkomt.
    # Deel alle elementen in de matrix 'element wise' door de grootste waarde in deze matrix.

    max_val = np.max(X)

    scaled_X = X / max_val

    return scaled_X


# OPGAVE 1c
def build_model():
    # Deze methode maakt het keras-model dat we gebruiken voor de classificatie van de mnist
    # dataset. Je hoeft deze niet abstract te maken, dus je kunt er van uitgaan dat de input
    # layer van dit netwerk alleen geschikt is voor de plaatjes in de opgave (wat is de 
    # dimensionaliteit hiervan?).
    # Maak een model met een input-laag, een volledig verbonden verborgen laag en een softmax
    # output-laag. Compileer het netwerk vervolgens met de gegevens die in opgave gegeven zijn
    # en retourneer het resultaat.

    # Het staat je natuurlijk vrij om met andere settings en architecturen te experimenteren.

    (X_train, y_train),(X_test, y_test) = keras.datasets.mnist.load_data()

    print("De dimnsionaliteit van de input layer is: " + str(X_train.shape))     # 60000 images van 28x28 pixels

    model = keras.Sequential()
    #https://keras.io/api/layers/activations/
    model.add([
        keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),  #reshapen naar een 1D array
        keras.layers.Dense(128, activation='relu'),                              #128 nodes in de hidden layer, relu als activatie functie
        keras.layers.Dense(10, activation='softmax')                             #10 mogelijkheden voor de output, softmax als activatie functie
    ])

    model.compile(optimizer='adam',                                              # optimizer past de gewichten aan om loss te minimaliseren
                    loss='sparse_categorical_crossentropy',                      # loss functie om het verschil tussen de voorspelde en echte waarden te berekenen
                    metrics=['accuracy'])                                        # metrics om de accuracy in te zien

    return model


# OPGAVE 2a
def conf_matrix(labels, pred):
    # Retourneer de econfusion matrix op basis van de gegeven voorspelling (pred) en de actuele
    # waarden (labels). Check de documentatie van tf.math.confusion_matrix:
    # https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix

    return tf.math.confusion_matrix(labels, pred)

    

# OPGAVE 2b
def conf_els(conf, labels): 
    # Deze methode krijgt een confusion matrix mee (conf) en een set van labels. Als het goed is, is 
    # de dimensionaliteit van de matrix gelijk aan len(labels) × len(labels) (waarom?). Bereken de 
    # waarden van de TP, FP, FN en TN conform de berekening in de opgave. Maak vervolgens gebruik van
    # de methodes zip() en list() om een list van len(labels) te retourneren, waarbij elke tupel 
    # als volgt is gedefinieerd:

    #     (categorie:string, tp:int, fp:int, fn:int, tn:int)
 
    # Check de documentatie van numpy diagonal om de eerste waarde te bepalen.
    # https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html

    print("De dimensionaliteit van de matrix is: " + str(conf.shape))
    print("Dit is gelijk aan het kwadraat van het aantal labels: " + str(len(labels) * len(labels)))

    tp = np.diagonal(conf)             #tp: diagonaal van de matrix
    fp = np.sum(conf, axis=0) - tp     #fp: som kolommen - tp                           axis 0 = kolommen van de cm
    fn = np.sum(conf, axis=1) - tp     #fn: som rijen - tp                              axis 1 = rijen van de cm
    tn = np.sum(conf) - tp - fp - fn   #tn: som van de hele matrix - tp - fp - fn

    return list(zip(labels, tp, fp, fn, tn)) #zip de lijsten tot een lijst van tuples

# OPGAVE 2c
def conf_data(metrics):
    # Deze methode krijgt de lijst mee die je in de vorige opgave hebt gemaakt (dus met lengte len(labels))
    # Maak gebruik van een list-comprehension om de totale tp, fp, fn, en tn te berekenen en 
    # bepaal vervolgens de metrieken die in de opgave genoemd zijn. Retourneer deze waarden in de
    # vorm van een dictionary (de scaffold hiervan is gegeven).

    # VERVANG ONDERSTAANDE REGELS MET JE EIGEN CODE
    
    tp_sum = sum(metric[1] for metric in metrics)
    fp_sum = sum(metric[2] for metric in metrics)
    fn_sum = sum(metric[3] for metric in metrics)
    tn_sum = sum(metric[4] for metric in metrics)

    # BEREKEN DE EVALUATIEMETRIEKEN
    true_positive_rate = tp_sum / (tp_sum + fn_sum)
    positive_predictive_value = tp_sum / (tp_sum + fp_sum)

    true_negative_rate = tn_sum / (tn_sum + fp_sum)
    fpr = fp_sum / (tn_sum + fp_sum) #fpr staat v

    # RETOURNEER DE METRIEKEN IN EEN DICTIONARY
    return {'TPR': tpr, 'PPV': ppv, 'TNR': tnr, 'FPR': fpr}
