import pandas as pd
import numpy as np
import gym
from gym import spaces
from sklearn.preprocessing import normalize

class EntityAlignmentEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, kge_model, args, links=None, is_test=False):
        super(EntityAlignmentEnvironment, self).__init__()
        
        self.args = args
        self.kgs = args.kgs
        self.kge_model = kge_model
        self.is_test=is_test
        self.candidate_num = args.candidate_num
        self.skip_rate = args.skip_rate
        self.seed(12345)
        
        if links is None:
            self.all_links = np.array(self.kgs.train_links)
        else:
            self.all_links = np.array(links)
        
        self.reset(hard=True)
        
    def seed(self, seed):
        # ensure reproducibility
        np.random.seed(seed)
    
    def eval_embeddings(self):
        kgs = self.kgs
        model = self.kge_model
        if not self.is_test:
            all_links = np.concatenate([kgs.train_links, kgs.valid_links, kgs.test_links])
        else:
            all_links = self.all_links
            
            
        self.loc2idx1 = pd.Series(all_links[:, 0], index=range(len(all_links)))
        self.loc2idx2 = pd.Series(all_links[:, 1], index=range(len(all_links)))
        
        
        if self.args.mapping:
            embeds = model.ent_embeds.eval(session=model.session)
            mapping = model.mapping_mat_1.eval(session=model.session)
            embeds[kgs.kg1.entities_list] = np.matmul(embeds[kgs.kg1.entities_list], mapping)
        else:
            embeds = model.ent_embeds.eval(session=model.session)
        
        embeds = normalize(embeds)
            
            
        embeds1 = embeds[all_links[:, 0]]
        embeds2 = embeds[all_links[:, 1]]

        
        # slice the rows to keep only given source entities
        return np.matmul(embeds1, embeds2.T)[:len(self.all_links)], embeds
                                         
    
    def reset_candidates(self):
        sim_mat, embeds = self.eval_embeddings()
        # select top-k entities as candidates
        candidate_ents = np.argpartition(-sim_mat, 
                                      self.candidate_num, axis=-1)[:, :self.candidate_num].flatten()
        
        source_ents = np.repeat(np.arange(sim_mat.shape[0]).reshape(-1, 1), 
                                self.candidate_num, 
                                axis=-1).flatten()
        
        assert source_ents.shape == candidate_ents.shape
        
        
        
        oppenent_ents = self.loc2idx2[candidate_ents].values.reshape([-1, self.candidate_num])
        candidate_probs = sim_mat[source_ents, candidate_ents]
        
        candidate_pairs = np.stack([source_ents, candidate_ents], axis=-1)
        
        candidate_pair_label = (candidate_pairs[:, 0] == candidate_pairs[:, 1])
        
        # calculate difficulty
        candidate_pair_probs = candidate_probs.flatten()
        max_diff_pair_probs =  candidate_probs.max(axis=-1) - candidate_probs
        
        candidate_pair_score = candidate_pair_label * max_diff_pair_probs + (1-candidate_pair_label) * (1-max_diff_pair_probs)
        
        # rescale to [0, 1]
        candidate_pair_score -= candidate_pair_score.min()
        candidate_pair_score /= candidate_pair_score.max(axis=-1) 
        
        candidate_df = pd.DataFrame({'source_ent':candidate_pairs[:, 0], 
                      'target_ent':candidate_pairs[:, 1],
                      'label':candidate_pair_label,
                      'score':candidate_pair_score,
                      'prob':candidate_pair_probs})
        
        candidate_df.sort_values(by='prob', ascending=False, inplace=True)
        
        candidate_df['source_ent_idx'] = self.loc2idx1.loc[candidate_df.source_ent.values].values
        candidate_df['target_ent_idx'] = self.loc2idx2.loc[candidate_df.target_ent.values].values
        candidate_df['oppenent_idx'] = candidate_df[['source_ent','target_ent_idx']].apply(lambda x: [o for o in  oppenent_ents[x.source_ent] if o!=x.target_ent_idx], axis=1)
    
        self.candidate_df = candidate_df
        self.embeds = embeds
        return candidate_df
        
        
    def reset(self, hard=False, random=False):
        
        self.matched_src, self.matched_tgt = [], []
        self.cursor = 0
        self.done = False
        self.skip_rate *= self.args.skip_discount_rate
        
        if hard:
            self.reset_candidates()
            self.skip_rate = self.args.skip_rate + 0.
        
        if random:
            self.candidate_df = self.candidate_df.sample(frac=1.0)

        return self._next_observation()
        
        
    
    def _next_observation(self):
        while self.cursor< len(self.candidate_df):
            candidate_pair = self.candidate_df.iloc[self.cursor]
            self.cursor += 1
            
            skip_rate = max(self.args.min_skip_rate, self.skip_rate*candidate_pair.score)
            skip = np.random.choice([0, 1], p=[1-skip_rate, skip_rate])
            if (not self.is_test) and skip:
                continue
            
            if (candidate_pair.target_ent not in self.matched_tgt) & (candidate_pair.source_ent not in self.matched_src):
                self.candidate_pair = candidate_pair
                return candidate_pair
            
        # terminal of one episode
        self.done = True
        self.candidate_pair = self.candidate_df.iloc[-1]
        return self.candidate_pair
    
    def step(self, action):
        if self.done:
            return self.candidate_pair, 0., self.done, {}
        
        reward = self._take_action(action)
        next_pair = self._next_observation()
        
        return next_pair, float(reward), self.done, {}
    
    def _take_action(self, action):
        label = self.candidate_pair.target_ent == self.candidate_pair.source_ent
        
        # True mismatch -- reward = 0
        if label == 0 and action == 0:
            score = 0
        # False mismatch -- reward = -1
        if label == 0 and action == 1:
            score = 0
        # False match -- severe penalty -10
        if label == 1 and action == 0:
            score = -10
        # True match -- reward = 1
        if label == 1 and action == 1:
             score = 1
        
        # if the agent believes matched
        if action:
            self.matched_src.append(self.candidate_pair.source_ent)
            self.matched_tgt.append(self.candidate_pair.target_ent)
        
        return score