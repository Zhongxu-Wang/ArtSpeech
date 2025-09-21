import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self,):
        pass
    def build_from_path(self):

        print("Processing Data ...")
        all_data = [[] for i in range(10)]

        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        for wav_name in tqdm(os.listdir("EMA")):
            ret = self.process_utterance(wav_name)
            if (ret == None):
                continue
            EMA, pitch, energy= ret
            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape((-1, 1)))
            for i in range(10):
                all_data[i] += list(EMA[i])

        EMA_mean = [float(np.mean(l)) for l in all_data]
        EMA_std = [float(np.std(l)) for l in all_data]

        pitch_mean = pitch_scaler.mean_[0]
        pitch_std = pitch_scaler.scale_[0]
        energy_mean = energy_scaler.mean_[0]
        energy_std = energy_scaler.scale_[0]

        with open("stats.json", "w") as f:
            stats = {
                "EMA": [EMA_mean,EMA_std],
                "pitch": [float(pitch_mean),float(pitch_std),],
                "energy": [float(energy_mean),float(energy_std),],}
            f.write(json.dumps(stats))
        return 0

    def process_utterance(self, basename):
        EMA = np.load("EMA/"+basename)
        energy = np.load("energy/"+basename)
        F0 = np.load("F0/"+basename)
        return (self.remove_outlier_EMA(EMA),
                self.remove_outlier(F0),
                self.remove_outlier(energy),)

    def remove_outlier_EMA(self, values):
        ans = []
        for line in values:
            p25 = np.percentile(line, 25)
            p75 = np.percentile(line, 75)
            lower = p25 - 1.5 * (p75 - p25)
            upper = p75 + 1.5 * (p75 - p25)
            normal_indices = np.logical_and(line > lower, line < upper)
            ans.append(line[normal_indices])
        return ans

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]


if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.build_from_path()
