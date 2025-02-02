import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# ==== OPGAVE 1 ====
def plot_number(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

    t = np.reshape(nrVector, (20, 20), order='F')
    plt.imshow(t, cmap='gray')
    plt.show()

# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.

    sig = 1 / (1 + np.exp(-z))

    return sig


# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m×x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0,0, ... 0] zijn, als
    # y_i=10, dan is regel i in de matrix [0,0,...1] (in dit geval is de breedte
    # van de matrix 10 (0-9), maar de methode moet werken voor elke waarde van 
    # y en m

    x = len(y)
    y_i = y[:, 0]
    
    cols = y_i - 1
    rows = np.arange(m)

    y_vec = csr_matrix((np.ones(x), (rows, cols)))

    y_vec_array = y_vec.toarray()
    
    return y_vec_array

# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predict_number(Theta1, Theta2, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta1 en Theta2. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta1 en Theta2 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk. 

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.

    m = X.shape[0]

    #Stap 1 Toevoegen van een bias aan de input
    a1 = np.hstack((np.ones((m, 1)), X))

    #Stap 2 Gewichten vermenigvuldigen met de input en sigmoid toepassen
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)

    #Stap 3 Toevoegen van een bias aan de verborgen laag
    m = a2.shape[0]
    a2 = np.hstack((np.ones((m, 1)), a2))

    #Stap 4 Gewichten vermenigvuldigen met de input en sigmoid toepassen
    z3 = np.dot(a2, Theta2.T)
    result = sigmoid(z3)

    return result


# ===== deel 2: =====
def compute_cost(Theta1, Theta2, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta1 en Theta2) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een matrix. 

    m = X.shape[0]

    y_matrix = get_y_matrix(y, m)

    h = predict_number(Theta1, Theta2, X)

    cost_matrix = -y_matrix * np.log(h) - (1 - y_matrix) * np.log(1 - h)

    cost = np.sum(cost_matrix) / m

    return cost


# ==== OPGAVE 3a ====
def sigmoid_gradient(z): 
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Zorg ervoor dat deze werkt met
    # scalaire waarden en met vectoren.

    return sigmoid(z) * (1 - sigmoid(z))


# ==== OPGAVE 3b ====
def nn_check_gradients(Theta1, Theta2, X, y): 
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

    Delta1 = np.zeros(Theta1.shape)
    Delta2 = np.zeros(Theta2.shape)
    m = X.shape[0]

    y_matrix = get_y_matrix(y, m)

    for i in range(m):
        # Stap 1 Forward propagation
        a1 = np.hstack(([1], X[i]))
        z2 = np.dot(a1, Theta1.T)
        a2 = sigmoid(z2)
        a2 = np.hstack(([1], a2))
        z3 = np.dot(a2, Theta2.T)
        a3 = sigmoid(z3)

        # Stap 2: Bereken de fout (delta) voor elke output node
        delta3 = a3 - y_matrix[i]

        # Stap 3 Bereken de fout (delta) voor elke node in verborgen laag
        delta2 = np.dot(delta3, Theta2) * sigmoid_gradient(np.hstack(([1], z2)))
        delta2 = delta2[1:]

        # Stap 4 Bijdrage optellen bij de totale bijdrage van elke node
        Delta1 += np.outer(delta2, a1)
        Delta2 += np.outer(delta3, a2)

    # Stap 5 Gemiddelde nemen van de bijdrage van elke node
    delta1_grad = Delta1 / m
    delta2_grad = Delta2 / m

    return delta1_grad, delta2_grad

    # === Resultaten ===
    #iteraties | accuratessse
    # 10       |   80.16%
    # 30       |   93.64%
    # 50       |   96.76%
    # 100      |   99,44%