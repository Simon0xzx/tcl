"""PEARL and PEARLWorker in Pytorch.

Code is adapted from https://github.com/katerakelly/oyster.
"""

import copy
import pickle
import akro
from dowel import logger
import numpy as np
import torch
import torch.nn.functional as F
from dowel import tabular


from garage import InOutSpec, TimeStep, EnvSpec, StepType
from garage.experiment import MetaEvaluator
from garage.np.algos import MetaRLAlgorithm
from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker
from garage.torch import global_device
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import ContextConditionedPolicy


class PEARLSAC2(MetaRLAlgorithm):
    """A PEARL model based on https://arxiv.org/abs/1903.08254.

    PEARL, which stands for Probablistic Embeddings for Actor-Critic
    Reinforcement Learning, is an off-policy meta-RL algorithm. It is built
    on top of SAC using two Q-functions and a value function with an addition
    of an inference network that estimates the posterior :math:`q(z \| c)`.
    The policy is conditioned on the latent variable Z in order to adpat its
    behavior to specific tasks.

    Args:
        env (list[GarageEnv]): Batch of sampled environment updates(EnvUpdate),
            which, when invoked on environments, will configure them with new
            tasks.
        policy_class (garage.torch.policies.Policy): Context-conditioned policy
            class.
        encoder_class (garage.torch.embeddings.ContextEncoder): Encoder class
            for the encoder in context-conditioned policy.
        inner_policy (garage.torch.policies.Policy): Policy.
        qf (torch.nn.Module): Q-function.
        vf (torch.nn.Module): Value function.
        num_train_tasks (int): Number of tasks for training.
        num_test_tasks (int): Number of tasks for testing.
        latent_dim (int): Size of latent context vector.
        encoder_hidden_sizes (list[int]): Output dimension of dense layer(s) of
            the context encoder.
        test_env_sampler (garage.experiment.SetTaskSampler): Sampler for test
            tasks.
        policy_lr (float): Policy learning rate.
        qf_lr (float): Q-function learning rate.
        vf_lr (float): Value function learning rate.
        context_lr (float): Inference network learning rate.
        policy_mean_reg_coeff (float): Policy mean regulation weight.
        policy_std_reg_coeff (float): Policy std regulation weight.
        policy_pre_activation_coeff (float): Policy pre-activation weight.
        soft_target_tau (float): Interpolation parameter for doing the
            soft target update.
        kl_lambda (float): KL lambda value.
        optimizer_class (callable): Type of optimizer for training networks.
        use_information_bottleneck (bool): False means latent context is
            deterministic.
        use_next_obs_in_context (bool): Whether or not to use next observation
            in distinguishing between tasks.
        meta_batch_size (int): Meta batch size.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        num_tasks_sample (int): Number of random tasks to obtain data for each
            iteration.
        num_steps_prior (int): Number of transitions to obtain per task with
            z ~ prior.
        num_steps_posterior (int): Number of transitions to obtain per task
            with z ~ posterior.
        num_extra_rl_steps_posterior (int): Number of additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        embedding_mini_batch_size (int): Number of transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        max_path_length (int): Maximum path length.
        discount (float): RL discount factor.
        replay_buffer_size (int): Maximum samples in replay buffer.
        reward_scale (int): Reward scale.
        update_post_train (int): How often to resample context when obtaining
            data during training (in trajectories).

    """

    # pylint: disable=too-many-statements
    def __init__(self,
                 env,
                 inner_policy,
                 qf1,
                 qf2,
                 sampler,
                 num_train_tasks,
                 num_test_tasks,
                 latent_dim,
                 encoder_hidden_sizes,
                 test_env_sampler,
                 policy_class=ContextConditionedPolicy,
                 encoder_class=MLPEncoder,
                 policy_lr=3E-4,
                 qf_lr=3E-4,
                 context_lr=3E-4,
                 policy_mean_reg_coeff=1E-3,
                 policy_std_reg_coeff=1E-3,
                 policy_pre_activation_coeff=0.,
                 soft_target_tau=0.005,
                 kl_lambda=.1,
                 fixed_alpha=None,
                 target_entropy=None,
                 initial_log_entropy=0.,
                 optimizer_class=torch.optim.Adam,
                 use_information_bottleneck=True,
                 use_next_obs_in_context=False,
                 meta_batch_size=64,
                 num_steps_per_epoch=1000,
                 num_initial_steps=100,
                 num_tasks_sample=100,
                 num_steps_prior=100,
                 num_steps_posterior=0,
                 num_extra_rl_steps_posterior=100,
                 batch_size=1024,
                 embedding_batch_size=1024,
                 embedding_mini_batch_size=1024,
                 max_path_length=500,
                 discount=0.99,
                 replay_buffer_size=1000000,
                 single_alpha = False,
                 reward_scale=1,
                 update_post_train=1):

        self._env = env
        self._qf1 = qf1
        self._qf2 = qf2
        # use 2 target q networks
        self._target_qf1 = copy.deepcopy(self._qf1)
        self._target_qf2 = copy.deepcopy(self._qf2)

        self._num_train_tasks = num_train_tasks
        self._num_test_tasks = num_test_tasks
        self._latent_dim = latent_dim

        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._context_lr = context_lr
        self._policy_mean_reg_coeff = policy_mean_reg_coeff
        self._policy_std_reg_coeff = policy_std_reg_coeff
        self._policy_pre_activation_coeff = policy_pre_activation_coeff
        self._soft_target_tau = soft_target_tau
        self._kl_lambda = kl_lambda
        self._use_information_bottleneck = use_information_bottleneck
        self._use_next_obs_in_context = use_next_obs_in_context
        self._meta_batch_size = meta_batch_size
        self._num_steps_per_epoch = num_steps_per_epoch
        self._num_initial_steps = num_initial_steps
        self._num_tasks_sample = num_tasks_sample
        self._num_steps_prior = num_steps_prior
        self._num_steps_posterior = num_steps_posterior
        self._num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self._batch_size = batch_size
        self._embedding_batch_size = embedding_batch_size
        self._embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self._discount = discount
        self._replay_buffer_size = replay_buffer_size
        self._reward_scale = reward_scale
        self._update_post_train = update_post_train
        self._task_idx = None
        self._is_resuming = False
        self._sampler = sampler
        self._optimizer_class = optimizer_class
        self._single_alpha = single_alpha

        worker_args = dict(deterministic=True, accum_context=True)
        self._evaluator = MetaEvaluator(test_task_sampler=test_env_sampler,
                                        worker_class=PEARLSAC2Worker,
                                        worker_args=worker_args,
                                        n_test_tasks=num_test_tasks)
        env_spec = env[0]()
        encoder_spec = self.get_env_spec(env_spec, latent_dim, 'encoder')
        encoder_in_dim = int(np.prod(encoder_spec.input_space.shape))
        encoder_out_dim = int(np.prod(encoder_spec.output_space.shape))
        context_encoder = encoder_class(input_dim=encoder_in_dim,
                                        output_dim=encoder_out_dim,
                                        hidden_sizes=encoder_hidden_sizes)

        # Automatic entropy coefficient tuning
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._initial_log_entropy = initial_log_entropy
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                self._target_entropy = -np.prod(
                    env_spec.action_space.shape).item()
            if self._single_alpha:
                self._log_alpha = torch.Tensor(
                    [self._initial_log_entropy]).requires_grad_()
            else:
                self._log_alpha = torch.Tensor([
                                                   self._initial_log_entropy] * self._num_train_tasks).requires_grad_()
            self._alpha_optimizer = self._optimizer_class([self._log_alpha],
                                                          lr=self._policy_lr)
        else:
            if self._single_alpha:
                self._log_alpha = torch.Tensor([self._fixed_alpha]).log()
            else:
                self._log_alpha = torch.Tensor(
                    [self._fixed_alpha] * self._num_train_tasks).log()

        self._policy = policy_class(
            latent_dim=latent_dim,
            context_encoder=context_encoder,
            policy=inner_policy,
            use_information_bottleneck=use_information_bottleneck,
            use_next_obs=use_next_obs_in_context)

        # buffer for training RL update
        self._replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(num_train_tasks)
        }
        self._context_replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self._policy_optimizer = optimizer_class(
            self._policy.networks[1].parameters(),
            lr=self._policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self._qf1.parameters(),
            lr=self._qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self._qf2.parameters(),
            lr=self._qf_lr,
        )
        self.context_optimizer = optimizer_class(
            self._policy.networks[0].parameters(),
            lr=self._context_lr,
        )

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        data = self.__dict__.copy()
        del data['_replay_buffers']
        del data['_context_replay_buffers']
        return data

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

        """
        self.__dict__.update(state)
        self._replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }

        self._context_replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }
        self._is_resuming = True

    def update_env(self, env, evaluator, num_train_tasks, num_test_tasks):
        print("Updating environments")
        self._env = env
        self._evaluator = evaluator
        self._num_train_tasks = num_train_tasks
        self._num_test_tasks = num_test_tasks
        # buffer for training RL update
        self._replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self._context_replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(num_train_tasks)
        }
        self._task_idx = 0
        print("Updated with new envipickleronment setup")
        self._policy_optimizer = torch.optim.Adam(
            self._policy.networks[1].parameters(),
            lr=3E-4,
        )
        self.qf1_optimizer = torch.optim.Adam(
            self._qf1.parameters(),
            lr=3E-4,
        )
        self.qf2_optimizer = torch.optim.Adam(
            self._qf2.parameters(),
            lr=3E-4,
        )
        self.vf_optimizer = torch.optim.Adam(
            self._vf.parameters(),
            lr=3E-4,
        )
        self.context_optimizer = torch.optim.Adam(
            self._policy.networks[0].parameters(),
            lr=3E-4,
        )

        print('Reset optimizer state')

    def fill_expert_traj(self, expert_traj_dir):
        print("Filling Expert trajectory to replay buffer")
        from os import listdir
        from os.path import isfile, join
        expert_traj_paths = [join(expert_traj_dir, f) for f in
                             listdir(expert_traj_dir) if
                             isfile(join(expert_traj_dir, f))]
        expert_trajs = []
        for exp_path in expert_traj_paths:
            with open(exp_path, 'rb') as handle:
                data = pickle.load(handle)
                expert_trajs.append(data)

        for path in expert_trajs:
            p = {
                'observations': path['observations'],
                'actions': path['actions'],
                'rewards': path['rewards'].reshape(-1, 1),
                'next_observations': path['next_observations'],
                'dones': path['dones'].reshape(-1, 1)
            }
            self._replay_buffers[self._task_idx].add_path(p)
            self._context_replay_buffers[self._task_idx].add_path(p)

    def adapt_expert_traj(self, runner):
        """Obtain samples, train, and evaluate for each epoch.

                Args:
                    runner (LocalRunner): LocalRunner is passed to give algorithm
                        the access to runner.step_epochs(), which provides services
                        such as snapshotting and sampler control.

                """
        for _ in runner.step_epochs():
            logger.log('Adapting Policy {}...'.format(runner.step_itr))
            self._train_once()
            runner.step_itr += 1
            logger.log('Evaluating...')
            # evaluate
            self._policy.reset_belief()
            self._evaluator.evaluate(self)

    def reset(self):
        self._policy.reset_belief()

    def train(self, runner):
        """Obtain samples, train, and evaluate for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """
        for _ in runner.step_epochs():
            epoch = runner.step_itr / self._num_steps_per_epoch

            # obtain initial set of samples from all train tasks
            if epoch == 0 or self._is_resuming:
                for idx in range(self._num_train_tasks):
                    self._task_idx = idx
                    self._obtain_samples(runner, epoch,
                                         self._num_initial_steps, np.inf)
                    self._is_resuming = False

            # obtain samples from random tasks
            for _ in range(self._num_tasks_sample):
                idx = np.random.randint(self._num_train_tasks)
                self._task_idx = idx
                self._context_replay_buffers[idx].clear()
                # obtain samples with z ~ prior
                if self._num_steps_prior > 0:
                    self._obtain_samples(runner, epoch, self._num_steps_prior,
                                         np.inf)
                # obtain samples with z ~ posterior
                if self._num_steps_posterior > 0:
                    self._obtain_samples(runner, epoch,
                                         self._num_steps_posterior,
                                         self._update_post_train)
                # obtain extras samples for RL training but not encoder
                if self._num_extra_rl_steps_posterior > 0:
                    self._obtain_samples(runner,
                                         epoch,
                                         self._num_extra_rl_steps_posterior,
                                         self._update_post_train,
                                         add_to_enc_buffer=False)

            logger.log('Training...')
            # sample train tasks and optimize networks
            self._train_once()
            runner.step_itr += 1

            logger.log('Evaluating...')
            # evaluate
            self._policy.reset_belief()
            self._evaluator.evaluate(self)

    def _train_once(self):
        """Perform one iteration of training."""
        policy_loss_list = []
        qf_loss_list = []
        alpha_loss_list = []
        alpha_list = []
        for _ in range(self._num_steps_per_epoch):
            indices = np.random.choice(range(self._num_train_tasks),
                                       self._meta_batch_size)
            policy_loss, qf_loss, alpha_loss, alpha = self._optimize_policy(indices)
            policy_loss_list.append(policy_loss)
            qf_loss_list.append(qf_loss)
            alpha_loss_list.append(alpha_loss)
            alpha_list.append(alpha)

        with tabular.prefix('MetaTrain/Average/'):
            tabular.record('PolicyLoss',  np.average(np.array(policy_loss_list)))
            tabular.record('QfLoss', np.average(np.array(qf_loss_list)))
            tabular.record('AlphaLoss', np.average(np.array(alpha_loss_list)))
            tabular.record('AlphaLoss', np.average(np.array(alpha_loss_list)))
            tabular.record('Alpha', np.average(np.array(alpha_list)))

    def _optimize_policy(self, indices):
        """Perform algorithm optimizing.

        Args:
            indices (list): Tasks used for training.

        """
        num_tasks = len(indices)
        context = self._sample_context(indices)
        # clear context and reset belief of policy
        self._policy.reset_belief(num_tasks=num_tasks)
        # data shape is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self._sample_data(indices)

        # flatten out the task dimension
        t, b, _ = obs.size()
        batch_obs = obs.view(t * b, -1)
        batch_action = actions.view(t * b, -1)
        batch_next_obs = next_obs.view(t * b, -1)

        policy_outputs, task_z = self._policy(next_obs, context)
        new_next_actions, policy_mean, policy_log_std, log_pi, pre_tanh = policy_outputs

        # ===== Critic Objective =====
        with torch.no_grad():
            alpha = self._get_log_alpha(indices).exp()
        q1_pred = self._qf1(torch.cat([batch_obs, batch_action], dim=1), task_z)
        q2_pred = self._qf2(torch.cat([batch_obs, batch_action], dim=1), task_z)
        target_q_values = torch.min(self._target_qf1(torch.cat([batch_next_obs, new_next_actions], dim=1), task_z),
            self._target_qf2(torch.cat([batch_next_obs, new_next_actions], dim=1), task_z)).flatten() - (alpha * log_pi.flatten())

        rewards_flat = rewards.view(self._batch_size * num_tasks, -1).flatten()
        rewards_flat = rewards_flat * self._reward_scale
        terms_flat = terms.view(self._batch_size * num_tasks, -1).flatten()

        with torch.no_grad():
            q_target = rewards_flat + ((1. - terms_flat) * self._discount) * target_q_values

        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)
        qf_loss = qf1_loss + qf2_loss

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self._use_information_bottleneck:
            kl_div = self._policy.compute_kl_div()
            kl_loss = self._kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # Optimize Q network and context encoder
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # ===== Actor Objective =====
        policy_outputs, task_z = self._policy(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi, pre_tanh = policy_outputs
        # compute min Q on the new actions
        q1 = self._qf1(torch.cat([batch_obs, new_actions], dim=1), task_z.detach())
        q2 = self._qf2(torch.cat([batch_obs, new_actions], dim=1), task_z.detach())
        min_q = torch.min(q1, q2)
        # optimize policy
        policy_loss = ((alpha * log_pi) - min_q.flatten()).mean()
        mean_reg_loss = self._policy_mean_reg_coeff * (policy_mean**2).mean()
        std_reg_loss = self._policy_std_reg_coeff * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self._policy_pre_activation_coeff * ((pre_tanh_value**2).sum(dim=1).mean())
        policy_reg_loss = (mean_reg_loss + std_reg_loss + pre_activation_reg_loss)
        policy_loss += policy_reg_loss

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # ===== Temperature Objective =====
        if self._use_automatic_entropy_tuning:
            alpha = (self._get_log_alpha(indices)).exp()
            alpha_loss = (-alpha * (log_pi.detach() + self._target_entropy)).mean()
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
            alpha_avg_cpu = np.average(alpha.detach().cpu().numpy())
            alpha_loss_cpu = alpha_loss.detach().cpu().numpy()

        # ===== Update Target Network =====
        target_qfs = [self._target_qf1, self._target_qf2]
        qfs = [self._qf1, self._qf2]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self._soft_target_tau) + param.data * self._soft_target_tau)

        qf_loss_cpu = qf_loss.detach().cpu().numpy()
        policy_loss_cpu = policy_loss.detach().cpu().numpy()
        return policy_loss_cpu, qf_loss_cpu, alpha_loss_cpu, alpha_avg_cpu

    def _obtain_samples(self,
                        runner,
                        itr,
                        num_samples,
                        update_posterior_rate,
                        add_to_enc_buffer=True):
        """Obtain samples.

        Args:
            runner (LocalRunner): LocalRunner.
            itr (int): Index of iteration (epoch).
            num_samples (int): Number of samples to obtain.
            update_posterior_rate (int): How often (in trajectories) to infer
                posterior of policy.
            add_to_enc_buffer (bool): Whether or not to add samples to encoder
                buffer.

        """
        self._policy.reset_belief()
        total_samples = 0

        if update_posterior_rate != np.inf:
            num_samples_per_batch = (update_posterior_rate *
                                     self.max_path_length)
        else:
            num_samples_per_batch = num_samples

        while total_samples < num_samples:
            paths = runner.obtain_samples(itr, num_samples_per_batch,
                                          self._policy,
                                          self._env[self._task_idx])
            total_samples += sum([len(path['rewards']) for path in paths])

            for path in paths:
                terminations=np.array([step_type == StepType.TERMINAL for step_type in path['step_types']]).reshape(-1, 1)
                p = {
                    'observations': path['observations'],
                    'actions': path['actions'],
                    'rewards': path['rewards'].reshape(-1, 1),
                    'next_observations': path['next_observations'],
                    'dones': terminations
                }
                self._replay_buffers[self._task_idx].add_path(p)

                if add_to_enc_buffer:
                    self._context_replay_buffers[self._task_idx].add_path(p)

            if update_posterior_rate != np.inf:
                context = self._sample_context(self._task_idx)
                self._policy.infer_posterior(context)

    def _sample_data(self, indices):
        """Sample batch of training data from a list of tasks.

        Args:
            indices (list): List of task indices to sample from.

        Returns:
            torch.Tensor: Obervations, with shape :math:`(X, N, O^*)` where X
                is the number of tasks. N is batch size.
            torch.Tensor: Actions, with shape :math:`(X, N, A^*)`.
            torch.Tensor: Rewards, with shape :math:`(X, N, 1)`.
            torch.Tensor: Next obervations, with shape :math:`(X, N, O^*)`.
            torch.Tensor: Dones, with shape :math:`(X, N, 1)`.

        """
        # transitions sampled randomly from replay buffer
        initialized = False
        for idx in indices:
            batch = self._replay_buffers[idx].sample_transitions(
                self._batch_size)
            if not initialized:
                o = batch['observations'][np.newaxis]
                a = batch['actions'][np.newaxis]
                r = batch['rewards'][np.newaxis]
                no = batch['next_observations'][np.newaxis]
                d = batch['dones'][np.newaxis]
                initialized = True
            else:
                o = np.vstack((o, batch['observations'][np.newaxis]))
                a = np.vstack((a, batch['actions'][np.newaxis]))
                r = np.vstack((r, batch['rewards'][np.newaxis]))
                no = np.vstack((no, batch['next_observations'][np.newaxis]))
                d = np.vstack((d, batch['dones'][np.newaxis]))

        o = torch.as_tensor(o, device=global_device()).float()
        a = torch.as_tensor(a, device=global_device()).float()
        r = torch.as_tensor(r, device=global_device()).float()
        no = torch.as_tensor(no, device=global_device()).float()
        d = torch.as_tensor(d, device=global_device()).float()

        return o, a, r, no, d

    def _sample_context(self, indices):
        """Sample batch of context from a list of tasks.

        Args:
            indices (list): List of task indices to sample from.

        Returns:
            torch.Tensor: Context data, with shape :math:`(X, N, C)`. X is the
                number of tasks. N is batch size. C is the combined size of
                observation, action, reward, and next observation if next
                observation is used in context. Otherwise, C is the combined
                size of observation, action, and reward.

        """
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        initialized = False
        for idx in indices:
            batch = self._context_replay_buffers[idx].sample_transitions(
                self._embedding_batch_size)
            o = batch['observations']
            a = batch['actions']
            r = batch['rewards']
            context = np.hstack((np.hstack((o, a)), r))
            if self._use_next_obs_in_context:
                context = np.hstack((context, batch['next_observations']))

            if not initialized:
                final_context = context[np.newaxis]
                initialized = True
            else:
                final_context = np.vstack((final_context, context[np.newaxis]))

        final_context = torch.as_tensor(final_context,
                                        device=global_device()).float()
        if len(indices) == 1:
            final_context = final_context.unsqueeze(0)

        return final_context

    @property
    def policy(self):
        """Return all the policy within the model.

        Returns:
            garage.torch.policies.Policy: Policy within the model.

        """
        return self._policy

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return self._policy.networks + [self._policy, self._qf1, self._qf2,
                                        self._target_qf1, self._target_qf2]

    def get_exploration_policy(self):
        """Return a policy used before adaptation to a specific task.

        Each time it is retrieved, this policy should only be evaluated in one
        task.

        Returns:
            garage.Policy: The policy used to obtain samples that are later
                used for meta-RL adaptation.

        """
        return self._policy

    def adapt_policy(self, exploration_policy, exploration_trajectories):
        """Produce a policy adapted for a task.

        Args:
            exploration_policy (garage.Policy): A policy which was returned
                from get_exploration_policy(), and which generated
                exploration_trajectories by interacting with an environment.
                The caller may not use this object after passing it into this
                method.
            exploration_trajectories (garage.TrajectoryBatch): Trajectories to
                adapt to, generated by exploration_policy exploring the
                environment.

        Returns:
            garage.Policy: A policy adapted to the task represented by the
                exploration_trajectories.

        """
        total_steps = sum(exploration_trajectories.lengths)
        o = exploration_trajectories.observations
        a = exploration_trajectories.actions
        r = exploration_trajectories.rewards.reshape(total_steps, 1)
        ctxt = np.hstack((o, a, r)).reshape(1, total_steps, -1)
        context = torch.as_tensor(ctxt, device=global_device()).float()
        self._policy.infer_posterior(context)

        return self._policy

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        device = device or global_device()
        for net in self.networks:
            net.to(device)

        if self._use_automatic_entropy_tuning:
            if self._single_alpha:
                self._log_alpha = torch.cuda.FloatTensor(
                    [self._initial_log_entropy],
                    device=global_device()).requires_grad_()
            else:
                self._log_alpha = torch.cuda.FloatTensor(
                    [self._initial_log_entropy] * self._num_train_tasks,
                    device=global_device()).requires_grad_()
            self._alpha_optimizer = self._optimizer_class([self._log_alpha],
                                                          lr=self._policy_lr)
        else:
            if self._single_alpha:
                self._log_alpha = torch.cuda.FloatTensor([self._fixed_alpha],
                                                         device=global_device()).log()
            else:
                self._log_alpha = torch.cuda.FloatTensor(
                    [self._fixed_alpha] * self._num_train_tasks,
                    device=global_device()).log()

    @classmethod
    def augment_env_spec(cls, env_spec, latent_dim):
        """Augment environment by a size of latent dimension.

        Args:
            env_spec (garage.envs.EnvSpec): Environment specs to be augmented.
            latent_dim (int): Latent dimension.

        Returns:
            garage.envs.EnvSpec: Augmented environment specs.

        """
        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape))
        aug_obs = akro.Box(low=-1,
                           high=1,
                           shape=(obs_dim + latent_dim, ),
                           dtype=np.float32)
        aug_act = akro.Box(low=-1,
                           high=1,
                           shape=(action_dim, ),
                           dtype=np.float32)
        return EnvSpec(aug_obs, aug_act)

    @classmethod
    def get_env_spec(cls, env_spec, latent_dim, module):
        """Get environment specs of encoder with latent dimension.

        Args:
            env_spec (garage.envs.EnvSpec): Environment specs.
            latent_dim (int): Latent dimension.
            module (str): Module to get environment specs for.

        Returns:
            garage.envs.InOutSpec: Module environment specs with latent
                dimension.

        """
        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape))
        if module == 'encoder':
            in_dim = obs_dim + action_dim + 1
            out_dim = latent_dim * 2
        elif module == 'vf':
            in_dim = obs_dim
            out_dim = latent_dim
        in_space = akro.Box(low=-1, high=1, shape=(in_dim, ), dtype=np.float32)
        out_space = akro.Box(low=-1,
                             high=1,
                             shape=(out_dim, ),
                             dtype=np.float32)
        if module == 'encoder':
            spec = InOutSpec(in_space, out_space)
        elif module == 'vf':
            spec = EnvSpec(in_space, out_space)

        return spec


    def _get_log_alpha(self, indices):
        """Return the value of log_alpha.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: log_alpha. shape is (1, self.buffer_batch_size)

        """
        log_alpha = self._log_alpha
        one_hots = np.zeros((len(indices) * self._batch_size, self._num_train_tasks), dtype=np.float32)
        for i in range(len(indices)):
            one_hots[self._batch_size * i: self._batch_size * (i + 1), indices[i]] = 1
        one_hots = torch.as_tensor(one_hots, device=global_device())
        ret = torch.mm(one_hots, log_alpha.unsqueeze(0).t()).squeeze()
        return ret

class PEARLSAC2Worker(DefaultWorker):
    """A worker class used in sampling for PEARL.

    It stores context and resample belief in the policy every step.

    Args:
        seed (int): The seed to use to intialize random number generators.
        max_episode_length(int or float): The maximum length of episodes which
            will be sampled. Can be (floating point) infinity.
        worker_number (int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        deterministic (bool): If True, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.
        accum_context (bool): If True, update context of the agent.

    Attributes:
        agent (Policy or None): The worker's agent.
        env (Environment or None): The worker's environment.

    """

    def __init__(self,
                 *,
                 seed,
                 max_episode_length,
                 worker_number,
                 deterministic=False,
                 accum_context=False):
        self._deterministic = deterministic
        self._accum_context = accum_context
        self._episode_info = None
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)

    def start_episode(self):
        """Begin a new episode."""
        self._eps_length = 0
        self._prev_obs, self._episode_info = self.env.reset()

    def step_episode(self):
        """Take a single time-step in the current episode.

        Returns:
            bool: True iff the episode is done, either due to the environment
            indicating termination of due to reaching `max_episode_length`.

        """
        if self._eps_length < self._max_episode_length:
            a, agent_info = self.agent.get_action(self._prev_obs)
            if self._deterministic:
                a = agent_info['mean']
            a, agent_info = self.agent.get_action(self._prev_obs)
            es = self.env.step(a)
            self._observations.append(self._prev_obs)
            self._env_steps.append(es)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            self._eps_length += 1

            if self._accum_context:
                s = TimeStep.from_env_step(env_step=es,
                                           last_observation=self._prev_obs,
                                           agent_info=agent_info,
                                           episode_info=self._episode_info)
                self.agent.update_context(s)
            if not es.last:
                self._prev_obs = es.observation
                return False
        self._lengths.append(self._eps_length)
        self._last_observations.append(self._prev_obs)
        return True

    def rollout(self):
        """Sample a single episode of the agent in the environment.

        Returns:
            EpisodeBatch: The collected episode.

        """
        self.agent.sample_from_belief()
        self.start_episode()
        while not self.step_episode():
            pass
        return self.collect_episode()
