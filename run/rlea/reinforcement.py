from collections import namedtuple
import itertools

import pandas as pd
import numpy as np

EpisodeStats = namedtuple(
    "Stats", ["episode_lengths", "episode_rewards", "episode_alignments"])
Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"])

def generate_one_episode(i_episode, sess, env, estimator_policy, args, test=False):
    logger = args.logger
    state = env.reset(random=args.random)
    episode = []
    alignment_number = 0
    rewards = 0
    steps = 0
    for t in itertools.count():
        # Take a step
        action_probs = estimator_policy.predict(state, sess=sess, env=env)

        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        label = state.target_ent == state.source_ent
        alignment_number += (label == action) & label
        steps += 1 

        
        next_state, reward, done, _ = env.step(action)
        rewards+=reward
        
        # Keep track of the transition
        episode.append(Transition(
            state=state, action=action, reward=reward, next_state=next_state, done=done))


        if done:
            if test:
                logger.info("\rStep {} @ Episode {}/{} (Rewards: {}, Alignment Number: {}/{}, Hits@1: {:.3f})".format(
                steps, i_episode, args.num_episodes, rewards, alignment_number, len(env.all_links), alignment_number/len(env.all_links)))
            else:
                logger.info("\rStep {} @ Episode {}/{} (Rewards: {}, Alignment Number {})".format(
                steps, i_episode, args.num_episodes, rewards, alignment_number))
            break

        state = next_state

    return episode

def update_estimators(episode, sess, env, estimator_policy, estimator_value, discount_rate):
    # Go through the episode and make policy updates
    loss = 0
    for t, transition in enumerate(episode):
        randi = np.random.randint(0, 8)
        if randi == 0:
            # The return after this step
            total_return = sum(discount_rate**i * t.reward for i,
                               t in enumerate(episode[t:]))
            
            
            # Calculate baseline/advantage
            baseline_value = estimator_value.predict(
                transition.state, sess=sess, env=env)
            advantage = total_return - baseline_value

            # Update value estimator
            estimator_value.update(
                transition.state, total_return, sess=sess, env=env)
            # Update policy estimator
            loss += estimator_policy.update(
                transition.state, advantage, transition.action, sess=sess, env=env)

    return loss


def reinforce(sess, env, valid_env, test_env, args, estimator_policy, estimator_value):
    
    for i_episode in range(0, args.num_episodes):
        logger = args.logger
        logger.info('TRAIN EPISODE:')
        episode = generate_one_episode(i_episode, sess, env, estimator_policy, args)

        
        loss = update_estimators(episode, sess, env, estimator_policy,
                          estimator_value, args.discount_rate)
        logger.info('Update policy networks, Loss: %.3f' % loss)

        if (i_episode+1) % 20 == 0:
            logger.info('VLIDE EPISODE:')
            _ = generate_one_episode(i_episode, sess, valid_env, estimator_policy, args, test=True)
            

            logger.info('TEST EPISODE:')
            _ = generate_one_episode(i_episode, sess, test_env, estimator_policy, args, test=True)


    return