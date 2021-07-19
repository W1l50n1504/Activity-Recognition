from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats as ss

from com.core import *
from com.utility import *

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

filename1 = absPath_ + '/immagine/grafico.png'
filename2 = absPath_ + '/immagine/graficoDistribuzione.png'

mydpi = 96


class HiddeMarkovModels(BaseModel, ABC):

    def __init__(self, nSamples):
        """
        nel costruttore della classe viene caricato il dataset che verrà utilizzato
        :param nSamples: int, indica il numero di samples che devono essere utilizzati per la generazione degli HMM
        :param dataset_: string, indica il nome del dataset che verrà utilizzato dal modello
        """
        super.__init__()
        self.X, self.y = self.loadData()
        self.data = self.ds.toList()
        self.logdata = np.log(self.data)
        self.nSamples = nSamples

    def train(self):
        """
        Metodo di train in cui viene creato il modello e viene allenato sul dataset caricato durante il costruttore
        :return:
        """
        # effettua il fitting dei modelli sui dati caricati dal dataset
        print('Creazione del modello e fitting dei dati...')
        self.model = GaussianHMM(n_components=6, n_iter=1000).fit(np.reshape(self.logdata, [len(self.logdata), 1]))
        print('fine fitting')

        # classificazione di ogni osservazione come stato 1 o 2 (al momento riconosce solo un tipo di attività)
        print('creazione hidden states')
        self.hidden_states = self.model.predict(np.reshape(self.logdata, [len(self.logdata), 1]))
        print('fine creazione hidden states')

        # trova i parametri di un HMM Gaussiano
        self.mus = np.array(self.model.means_)
        self.sigmas = np.array(np.sqrt(np.array([np.diag(self.model.covars_[0]), np.diag(self.model.covars_[1])])))
        self.P = np.array(self.model.transmat_)

        # trova la log-likelihood di una HMM Gaussiana
        self.logProb = self.model.score(np.reshape(self.data, [len(self.data), 1]))

        # genera nSamples dagli HMM Gaussiani
        self.samples = self.model.sample(self.nSamples)

        # riorganizza i mu, i sigma e P in modo che la prima colonna contenga i lower mean (se non già presenti)
        if self.mus[0] > self.mus[1]:
            self.mus = np.flipud(self.mus)
            self.sigmas = np.flipud(self.sigmas)
            self.P = np.fliplr(np.flipud(self.P))
            self.hidden_states = 1 - self.hidden_states

    def plotScatter(self):
        """
        crea un grafico scatter in cui vengono plottati i dati del dataset (asse x = posizione del dato plottato nel dataset,
        asse y = valore registrato dal sensore stesso) e con il colore del punto viene indicato la categoria di appartenenza
        di quel dato secondo il sistema
        Infine salva il grafico
        :return:
        """
        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        xs = np.arange(len(self.logdata))

        masks = self.hidden_states == 0
        ax.scatter(xs[masks], self.logdata[masks], c='red', label='WalkingDwnStairs')

        masks = self.hidden_states == 1
        ax.scatter(xs[masks], self.logdata[masks], c='blue', label='WalkingUpstairs')

        masks = self.hidden_states == 2
        ax.scatter(xs[masks], self.logdata[masks], c='green', label='Sitting')

        masks = self.hidden_states == 3
        ax.scatter(xs[masks], self.logdata[masks], c='yellow', label='Standing')

        masks = self.hidden_states == 4
        ax.scatter(xs[masks], self.logdata[masks], c='orange', label='Walking')

        masks = self.hidden_states == 5
        ax.scatter(xs[masks], self.logdata[masks], c='black', label='Jogging')

        # decommentare per congiungere tutti i punti sul grafico
        # ax.plot(xs, self.logdata, c='k')

        ax.set_xlabel('Indice')
        ax.set_ylabel('Valore sensore')
        fig.subplots_adjust(bottom=0.2)
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
        fig.set_size_inches(800 / mydpi, 800 / mydpi)
        fig.savefig(filename1)
        fig.clf()

    def plotDistribution(self):
        """
        crea il grafico di distribuzione del modello utilizzando i valori mu e sigma dello stesso, teoricamente dovrebbero uscire delle gaussiane,
        una per ogni hidden state e infine una funzione che comprende tutte le gaussiane caricate
        :return:
        """
        # calcolo della distribuzione stazionaria
        eigenvals, eigenvecs = np.linalg.eig(np.transpose(self.P))
        one_eigval = np.argmin(np.abs(eigenvals - 1))
        pi = eigenvecs[:, one_eigval] / np.sum(eigenvecs[:, one_eigval])

        x_0 = np.linspace(self.mus[0] - 4 * self.sigmas[0], self.mus[0] + 4 * self.sigmas[0], 10000)
        fx_0 = pi[0] * ss.norm.pdf(x_0, self.mus[0], self.sigmas[0])

        x_1 = np.linspace(self.mus[1] - 4 * self.sigmas[1], self.mus[1] + 4 * self.sigmas[1], 10000)
        fx_1 = pi[1] * ss.norm.pdf(x_1, self.mus[1], self.sigmas[1])

        # x_2 = np.linspace(self.mus[2] - 4 * self.sigmas[2], self.mus[2] + 4 * self.sigmas[2], 10000)
        # fx_2 = pi[2] * ss.norm.pdf(x_2, self.mus[2], self.sigmas[2])

        # x_3= np.linspace(self.mus[3] - 4 * self.sigmas[3], self.mus[3] + 4 * self.sigmas[1], 10000)
        # fx_3 = pi[3] * ss.norm.pdf(x_3, self.mus[3], self.sigmas[3])

        # x_4= np.linspace(self.mus[4] - 4 * self.sigmas[4], self.mus[4] + 4 * self.sigmas[4], 10000)
        # fx_4 = pi[4] * ss.norm.pdf(x_4, self.mus[4], self.sigmas[4])

        # x_5 = np.linspace(self.mus[5] - 4 * self.sigmas[5], self.mus[5] + 4 * self.sigmas[1], 10000)
        # fx_5 = pi[5] * ss.norm.pdf(x_5, self.mus[5], self.sigmas[5])

        x = np.linspace(self.mus[0] - 4 * self.sigmas[0], self.mus[1] + 4 * self.sigmas[1], 10000)
        fx = pi[0] * ss.norm.pdf(x, self.mus[0], self.sigmas[0]) + pi[1] * ss.norm.pdf(x, self.mus[1], self.sigmas[1])

        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(self.logdata, color='k', alpha=0.5, density=True)
        l1, = ax.plot(x_0, fx_0, c='red', linewidth=2, label='WalkingDwnStairs Distn')
        l2, = ax.plot(x_1, fx_1, c='blue', linewidth=2, label='WalkingUpStairs Distn')
        # l3, = ax.plot(x_2, fx_2, c='green', linewidth=2, label='Sitting Distn')
        # l4, = ax.plot(x_3, fx_3, c='yellow', linewidth=2, label='Standing Distn')
        # l5, = ax.plot(x_4, fx_4, c='orange', linewidth=2, label='Walking Distn')
        # l6, = ax.plot(x_5, fx_5, c='black', linewidth=2, label='Jogging Distn')
        l7, = ax.plot(x, fx, c='cyan', linewidth=2, label='Combined Distn')

        fig.subplots_adjust(bottom=0.15)
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True)
        fig.set_size_inches(800 / mydpi, 800 / mydpi)
        fig.savefig(filename2)
        fig.clf()


if __name__ == '__main__':
    hmm = HiddeMarkovModels(100, trainset)
    hmm.train()
    hmm.plotScatter()
    hmm.plotDistribution()
