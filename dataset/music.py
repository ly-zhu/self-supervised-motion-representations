import random
import numpy as np
from .base import BaseDataset

class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.num_class = opt.num_class
        self.dataFlag = opt.dataFlag
        self.audShiftFlag = opt.audShiftFlag
        self.shiftRegressionFlag = opt.shiftRegressionFlag
        self.margin_dur = opt.margin_dur
        self.vid_dur = opt.vid_dur
        self.shift_dur = opt.shift_dur
        self.non_inter_dur = opt.non_inter_dur
        self.dataset = opt.dataset

    def __getitem__(self, index):
        N = self.num_mix

        global_frame_emb = np.squeeze(np.eye(self.num_class))

        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]
        frame_emb = [None for n in range(N)]
        classes = [None for n in range(N)]

        #if shift audio waveform
        if self.audShiftFlag:
            audios_shift = [None for n in range(N)]
            center_shift_frames = [0 for n in range(N)]
            #y_shift = [None for n in range(N)]
            y_shift_reg = [None for n in range(N)]
            y_random = [None for n in range(N)]

        # the first video
        infos[0] = self.list_sample[index]
        path_audioN = infos[0][1]
        music_category = int(infos[0][2])
        frame_emb[0] = np.array(global_frame_emb[music_category,:])
        classes[0] = music_category

        music_lib = []
        music_lib.append(music_category)

        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            second_music_category = music_category


            second_video_idx = random.randint(0, len(self.list_sample)-1)
            second_music_category = int(self.list_sample[second_video_idx][2])

            # if want load diff and same category data random equally
            dataFlag0_sign = False
            if self.dataFlag == 0:
                dataFlag0_sign = True
                diff_or_not = random.randint(0, 1)
                if diff_or_not:
                    self.dataFlag = 1
                else:
                    self.dataFlag = 2

            # if only want to load diff class data    
            if self.dataFlag == 1:
                while second_music_category in music_lib:
                    second_video_idx = random.randint(0, len(self.list_sample)-1)
                    second_music_category = int(self.list_sample[second_video_idx][2])
            # if only want to load same class data  
            if self.dataFlag == 2:
                while second_music_category not in music_lib:
                    second_video_idx = random.randint(0, len(self.list_sample)-1)
                    second_music_category = int(self.list_sample[second_video_idx][2])

            music_lib.append(second_music_category)


            infos[n] = self.list_sample[second_video_idx]
            frame_emb[n] = np.array(global_frame_emb[second_music_category,:])
            classes[n] = second_music_category

            if dataFlag0_sign:
                self.dataFlag = 0


        if N == 2:
            if music_category == second_music_category:
                match = 1
            else:
                match = 0

        idx_margin = max(
            int(self.margin_dur), (self.num_frames // 2) * self.stride_frames + 1)
        for n, infoN in enumerate(infos):
            path_frameN, path_audioN, _, count_framesN = infoN

            # if considering audio shifting
            if self.audShiftFlag:
                if self.split == 'train':
                    choices = []
                    # 6 + 1, 1 is the non-intersection margin we set
                    # start point in shift range is (0, total-max)
                    max_frame = int(self.num_frames + self.non_inter_dur)
                    select_start = random.randint(
                            idx_margin, int(count_framesN) - 1 - idx_margin - max_frame - self.num_frames)
                    # frames range (0, max)
                    frames1 = range(select_start, select_start + max_frame)

                    for frame1 in frames1:
                        for frame2 in reversed(frames1):
                            inv1 = range(frame1, frame1 + self.num_frames)
                            inv2 = range(frame2, frame2 + self.num_frames)
                            if len(set(inv1).intersection(inv2)) <= self.vid_dur-self.shift_dur:
                                choices.append([frame1, frame2])

                    idx = random.randint(0, len(choices)-1)
                    choices = np.array(choices)
                    start_frame_gt = choices[idx, 0]
                    start_frame_shift = choices[idx, 1]

                    # shifted direction and dur, when it is negative, shift to right side
                    y_shift_reg[n] = (start_frame_shift - start_frame_gt)*(1.0/self.fps)
                    center_frameN = int(start_frame_gt + self.num_frames/2)
                    center_shift_frameN = int(start_frame_shift + self.num_frames/2)
                else:
                    center_frameN = int(count_framesN) // 2
                    center_shift_frameN =  center_frameN + self.shift_dur
                    y_shift_reg[n] = 1.0/self.fps

                center_frames[n] = center_frameN + 1
                center_shift_frames[n] = center_shift_frameN + 1

                # randomly chose to shift or not, if idx=0 then shift, if idx=1 not shift
                y_random[n] = random.randint(0, 1)


            else:
                # if not considering audio shifting
                if self.split == 'train':
                    center_frameN = random.randint(
                        idx_margin, int(count_framesN) - idx_margin - 1)
                else:
                    center_frameN = int(count_framesN) // 2
                center_frames[n] = center_frameN + 1


            # absolute frame/audio paths
            data_path = '../../dataset/' + self.dataset + '/'
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(data_path + 'frames/' + path_frameN+'/{:06d}.jpg'.format(center_frames[n] + idx_offset))
            path_audios[n] = data_path + path_audioN
        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames(path_frames[n])
                if self.audShiftFlag:
                    center_timeN_shift = (center_shift_frames[n] - 0.5) / self.fps
                    audios_shift[n] = self._load_audio(path_audios[n], center_timeN_shift)
                center_timeN = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            if not self.audShiftFlag:
                mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)
            else:
                mags = self._n_stft(audios)
                mags_shift = self._n_stft(audios_shift)
        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            if self.audShiftFlag:
                frames, audios, mags, mags_shift = self.dummy_mix_frame_audio(N)
            else:
                mag_mix, mags, frames, audios, phase_mix = \
                    self.dummy_mix_data(N)


        if self.audShiftFlag:
            ret_dict = {'mags': mags, 'mags_shift': mags_shift, 'frames': frames, 'audios': audios, 'audios_shift': audios_shift, 'frame_emb': frame_emb, 'classes': classes, 'y_shift_reg': y_shift_reg, 'y_random': y_random}
        else:
            ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'audios': audios, 'mags': mags, 'frame_emb': frame_emb, 'classes': classes}
        if N == 2:
            ret_dict['match'] = match
        if self.split != 'train':
            if not self.audShiftFlag:
                ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos
        return ret_dict