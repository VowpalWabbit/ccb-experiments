#pragma once

int slot_dependent_ctr()
{
  auto vw_ccb = VW::initialize("--ccb_explore_adf --epsilon 0.2 -l 0.001 -q UA --power_t 0 --quiet");
  auto vw_cb = VW::initialize("--cb_explore_adf --epsilon 0.2 -l 0.001 --power_t 0 --quiet --cb_sample --quadratic UA");

  auto const NUM_USERS = 3;
  auto const NUM_ACTIONS = 5;
  auto const NUM_SLOTS = 3;
  auto const NUM_ITER = 1000000;

  std::vector<std::string> user_features = {"a", "b", "c"};
  std::vector<std::string> action_features = {"d", "e", "f", "ff", "fff"};

  auto cb_clicks_impressions = generate_clicks_impressions_store(NUM_ACTIONS, NUM_SLOTS, NUM_USERS);
  auto clicks_impressions = generate_clicks_impressions_store(NUM_ACTIONS, NUM_SLOTS, NUM_USERS);

  std::vector<std::map<std::vector<size_t>, std::vector<float>>> user_slot_action_probabilities;
  user_slot_action_probabilities.push_back(generate_slot_dependent_probs(NUM_ACTIONS, NUM_SLOTS,
                                                                         {{0.1f, 0.1f, 0.3f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.4f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.1f, 0.5f, 0.1f}}));
  user_slot_action_probabilities.push_back(generate_slot_dependent_probs(NUM_ACTIONS, NUM_SLOTS,
                                                                         {{0.2f, 0.1f, 0.1f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.01f, 0.1f, 0.5f}, {0.1f, 0.3f, 0.1f, 0.1f, 0.1f}}));
  user_slot_action_probabilities.push_back(generate_slot_dependent_probs(NUM_ACTIONS, NUM_SLOTS,
                                                                         {{0.1f, 0.3f, 0.1f, 0.1f, 0.4f}, {0.6f, 0.1f, 0.1f, 0.1f, 0.1f}, {0.1f, 0.2f, 0.1f, 0.6f, 0.1f}}));

  std::default_random_engine rd{0};
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> user_distribution(0, NUM_USERS - 1);
  std::uniform_real_distribution<float> click_distribution(0.0f, 1.0f);
  uint64_t merand_seed = 0;

  for (int i = 1; i <= NUM_ITER; i++)
  {
    auto chosen_user = user_distribution(eng);

    {
      // DO CCB
      auto ccb_ex_str = build_example_string_ccb(user_features[chosen_user], action_features, NUM_SLOTS);

      multi_ex ccb_ex_col;
      for (auto str : ccb_ex_str)
      {
        ccb_ex_col.push_back(VW::read_example(*vw_ccb, str));
      }

      vw_ccb->predict(ccb_ex_col);

      std::vector<std::tuple<size_t, float, float>> outcomes;
      auto decision_scores = ccb_ex_col[0]->pred.decision_scores;

      std::vector<size_t> actions_taken;
      for (auto s : decision_scores)
      {
        actions_taken.push_back(s[0].action);
      };

      std::get<1>(clicks_impressions[chosen_user][actions_taken])++;
      for (auto slot_id = 0; slot_id < decision_scores.size(); slot_id++)
      {
        auto &slot = decision_scores[slot_id];
        auto action_id = slot[0].action;
        auto prob_chosen = slot[0].score;
        auto prob_to_click = user_slot_action_probabilities[chosen_user][actions_taken][slot_id];

        if (click_distribution(eng) < prob_to_click)
        {
          std::get<0>(clicks_impressions[chosen_user][actions_taken])[slot_id]++;
          outcomes.emplace_back(action_id, -1.f, prob_chosen);
        }
        else
        {
          outcomes.emplace_back(action_id, 0.f, prob_chosen);
        }
      }
      as_multiline(vw_ccb->l)->finish_example(*vw_ccb, ccb_ex_col);

      auto learn_ex = build_example_string_ccb(user_features[chosen_user], action_features, NUM_SLOTS, outcomes);
      multi_ex learn_ex_col;
      for (auto str : learn_ex)
      {
        learn_ex_col.push_back(VW::read_example(*vw_ccb, str));
      }
      vw_ccb->learn(learn_ex_col);
      as_multiline(vw_ccb->l)->finish_example(*vw_ccb, learn_ex_col);
    }

    {
      std::vector<size_t> cb_actions_taken;
      std::vector<float> cb_probs;

      auto cb_ex_str = build_example_string_cb_no_slot(user_features[chosen_user], action_features);

      multi_ex cb_ex_col;
      for (auto str : cb_ex_str)
      {
        cb_ex_col.push_back(VW::read_example(*vw_cb, str));
      }

      vw_cb->predict(cb_ex_col);

      for (int slot_id = 0; slot_id < NUM_SLOTS; slot_id++)
      {
        auto action_score = cb_ex_col[0]->pred.a_s;
        cb_actions_taken.push_back(action_score[slot_id].action);
        cb_probs.push_back(action_score[slot_id].score);
      }

      as_multiline(vw_cb->l)->finish_example(*vw_cb, cb_ex_col);

      // Calculate reward for top action.
      std::vector<std::tuple<size_t, float, float>> cb_outcomes;
      for (auto slot_id = 0; slot_id < cb_actions_taken.size(); slot_id++)
      {
        auto prob_to_click = user_slot_action_probabilities[chosen_user][cb_actions_taken][slot_id];

        if (click_distribution(eng) < prob_to_click)
        {
          std::get<0>(cb_clicks_impressions[chosen_user][cb_actions_taken])[slot_id]++;
          cb_outcomes.emplace_back(cb_actions_taken[slot_id], -1.f, cb_probs[slot_id]);
        }
        else
        {
          cb_outcomes.emplace_back(cb_actions_taken[slot_id], 0.f, cb_probs[slot_id]);
        }
      }
      std::get<1>(cb_clicks_impressions[chosen_user][cb_actions_taken])++;

      // Learn from top action.
      auto learn_ex = build_example_string_cb_no_slot(user_features[chosen_user], action_features, cb_outcomes[0]);
      multi_ex cb_learn_ex_col;
      for (auto str : learn_ex)
      {
        cb_learn_ex_col.push_back(VW::read_example(*vw_cb, str));
      }
      vw_cb->learn(cb_learn_ex_col);
      as_multiline(vw_cb->l)->finish_example(*vw_cb, cb_learn_ex_col);
    }

    if (i % 5000 == 0)
    {
      // Clear terminal
      //std::cout << "\033[2J" << std::endl;
      //print_ctr(i, NUM_SLOTS, clicks_impressions);
      //std::cout << "============================================== \n --CB-- " << std::endl;
      //print_ctr(i, NUM_SLOTS, cb_clicks_impressions);

      print_ctr_as_csv(i, NUM_SLOTS, clicks_impressions, cb_clicks_impressions);
    }
  }

  //std::cout << "\033[2J" << std::endl;
  //print_ctr(NUM_ITER, NUM_SLOTS, clicks_impressions);
  //std::cout << "============================================== \n --CB-- " << std::endl;
  //print_ctr(NUM_ITER, NUM_SLOTS, cb_clicks_impressions);

  print_ctr_as_csv(NUM_ITER, NUM_SLOTS, clicks_impressions, cb_clicks_impressions);

  std::cout << "user,slot,action,count" << std::endl;
  for (auto user = 0; user < NUM_USERS; user++)
  {
    for (auto slot = 0; slot < NUM_SLOTS; slot++)
    {
      for (auto action = 0; action < NUM_ACTIONS; action++)
      {
        std::cout << user << "," << slot << "," << action << ","
                  << get_num_shown(user, slot, action, clicks_impressions) << std::endl;
      }
    }
  }

  std::cout << "\n\n\n"
            << std::endl;

  std::cout << "user,slot,action,count" << std::endl;
  for (auto user = 0; user < NUM_USERS; user++)
  {
    for (auto slot = 0; slot < NUM_SLOTS; slot++)
    {
      for (auto action = 0; action < NUM_ACTIONS; action++)
      {
        std::cout << user << "," << slot << "," << action << ","
                  << get_num_shown(user, slot, action, cb_clicks_impressions) << std::endl;
      }
    }
  }
  return 0;
}


int previous_action_dependent_ctr_3_slots_4_actions()
{
  auto vw_ccb = VW::initialize("--ccb_explore_adf --epsilon 0.2 -l 0.001 --power_t 0 --quiet");
  auto vw_cb = VW::initialize("--cb_explore_adf --epsilon 0.2 -l 0.001 --power_t 0 --quiet --cb_sample --quadratic UA");

  auto const NUM_USERS = 3;
  auto const NUM_ACTIONS = 4;
  auto const NUM_SLOTS = 3;
  auto const NUM_ITER = 1000000;

  std::vector<std::string> user_features = {"a", "b", "c"};
  std::vector<std::string> action_features = {"d", "e", "f", "ff"};
  std::vector<std::string> slot_features = {"h", "i", "j"};

  auto cb_clicks_impressions = generate_clicks_impressions_store(NUM_ACTIONS, NUM_SLOTS, NUM_USERS);
  auto clicks_impressions = generate_clicks_impressions_store(NUM_ACTIONS, NUM_SLOTS, NUM_USERS);

  // clang-format off
  std::vector<std::map<std::vector<size_t>,std::vector<float>>> user_slot_action_probabilities =
  {
    {
      {{0,1,2}, {0.01f, 0.3f, 0.8f}},
      {{1,0,2}, {0.01f, 0.01f, 0.01f}},
      {{2,0,1}, {0.01f, 0.01f, 0.01f}},
      {{0,2,1}, {0.01f, 0.01f, 0.01f}},
      {{1,2,0}, {0.01f, 0.01f, 0.01f}},
      {{2,1,0}, {0.01f, 0.01f, 0.01f}},
      {{0,1,3}, {0.01f, 0.01f, 0.01f}},
      {{1,0,3}, {0.01f, 0.01f, 0.01f}},
      {{3,0,1}, {0.01f, 0.01f, 0.01f}},
      {{0,3,1}, {0.01f, 0.01f, 0.01f}},
      {{1,3,0}, {0.01f, 0.8f, 0.01f}},
      {{3,1,0}, {0.01f, 0.01f, 0.01f}},
      {{0,2,3}, {0.01f, 0.01f, 0.01f}},
      {{2,0,3}, {0.01f, 0.01f, 0.01f}},
      {{3,0,2}, {0.01f, 0.01f, 0.01f}},
      {{0,3,2}, {0.01f, 0.01f, 0.01f}},
      {{2,3,0}, {0.01f, 0.01f, 0.01f}},
      {{3,2,0}, {0.01f, 0.01f, 0.01f}},
      {{1,2,3}, {0.01f, 0.01f, 0.01f}},
      {{2,1,3}, {0.01f, 0.01f, 0.01f}},
      {{3,1,2}, {0.01f, 0.01f, 0.01f}},
      {{1,3,2}, {0.01f, 0.01f, 0.01f}},
      {{2,3,1}, {0.01f, 0.01f, 0.01f}},
      {{3,2,1}, {0.01f, 0.01f, 0.01f}}
    },
    {
      {{0,1,2}, {0.01f, 0.01f, 0.01f}},
      {{1,0,2}, {0.01f, 0.01f, 0.01f}},
      {{2,0,1}, {0.01f, 0.01f, 0.01f}},
      {{0,2,1}, {0.01f, 0.01f, 0.01f}},
      {{1,2,0}, {0.2f, 0.01f, 0.01f}},
      {{2,1,0}, {0.01f, 0.01f, 0.8f}},
      {{0,1,3}, {0.01f, 0.01f, 0.01f}},
      {{1,0,3}, {0.01f, 0.01f, 0.01f}},
      {{3,0,1}, {0.01f, 0.01f, 0.01f}},
      {{0,3,1}, {0.01f, 0.01f, 0.01f}},
      {{1,3,0}, {0.01f, 0.01f, 0.01f}},
      {{3,1,0}, {0.01f, 0.01f, 0.01f}},
      {{0,2,3}, {0.01f, 0.01f, 0.01f}},
      {{2,0,3}, {0.01f, 0.01f, 0.01f}},
      {{3,0,2}, {0.01f, 0.01f, 0.01f}},
      {{0,3,2}, {0.01f, 0.01f, 0.01f}},
      {{2,3,0}, {0.01f, 0.01f, 0.01f}},
      {{3,2,0}, {0.01f, 0.01f, 0.01f}},
      {{1,2,3}, {0.01f, 0.01f, 0.01f}},
      {{2,1,3}, {0.01f, 0.01f, 0.01f}},
      {{3,1,2}, {0.01f, 0.01f, 0.01f}},
      {{1,3,2}, {0.01f, 0.01f, 0.01f}},
      {{2,3,1}, {0.01f, 0.01f, 0.01f}},
      {{3,2,1}, {0.01f, 0.01f, 0.01f}}
    },
    {
      {{0,1,2}, {0.01f, 0.01f, 0.01f}},
      {{1,0,2}, {0.01f, 0.01f, 0.01f}},
      {{2,0,1}, {0.01f, 0.01f, 0.01f}},
      {{0,2,1}, {0.01f, 0.01f, 0.01f}},
      {{1,2,0}, {0.01f, 0.01f, 0.01f}},
      {{2,1,0}, {0.01f, 0.01f, 0.01f}},
      {{0,1,3}, {0.01f, 0.01f, 0.01f}},
      {{1,0,3}, {0.01f, 0.01f, 0.01f}},
      {{3,0,1}, {0.01f, 0.01f, 0.01f}},
      {{0,3,1}, {0.01f, 0.01f, 0.01f}},
      {{1,3,0}, {0.01f, 0.01f, 0.01f}},
      {{3,1,0}, {0.01f, 0.01f, 0.8f}},
      {{0,2,3}, {0.01f, 0.01f, 0.01f}},
      {{2,0,3}, {0.01f, 0.01f, 0.01f}},
      {{3,0,2}, {0.01f, 0.01f, 0.01f}},
      {{0,3,2}, {0.01f, 0.01f, 0.01f}},
      {{2,3,0}, {0.01f, 0.01f, 0.01f}},
      {{3,2,0}, {0.01f, 0.01f, 0.01f}},
      {{1,2,3}, {0.01f, 0.01f, 0.01f}},
      {{2,1,3}, {0.3f, 0.01f, 0.01f}},
      {{3,1,2}, {0.01f, 0.01f, 0.01f}},
      {{1,3,2}, {0.01f, 0.01f, 0.01f}},
      {{2,3,1}, {0.01f, 0.01f, 0.01f}},
      {{3,2,1}, {0.01f, 0.01f, 0.01f}}
    }
};
  // clang-format on
  std::default_random_engine rd{0};
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> user_distribution(0, NUM_USERS - 1);
  std::uniform_real_distribution<float> click_distribution(0.0f, 1.0f);
  uint64_t merand_seed = 0;

  for (int i = 1; i <= NUM_ITER; i++)
  {
    auto chosen_user = user_distribution(eng);

    {
      // DO CCB
      auto ccb_ex_str = build_example_string_ccb(user_features[chosen_user], action_features, slot_features);

      multi_ex ccb_ex_col;
      for (auto str : ccb_ex_str)
      {
        ccb_ex_col.push_back(VW::read_example(*vw_ccb, str));
      }

      vw_ccb->predict(ccb_ex_col);

      std::vector<std::tuple<size_t, float, float>> outcomes;
      auto decision_scores = ccb_ex_col[0]->pred.decision_scores;

      std::vector<size_t> actions_taken;
      for (auto s : decision_scores)
      {
        actions_taken.push_back(s[0].action);
      };

      std::get<1>(clicks_impressions[chosen_user][actions_taken])++;
      for (auto slot_id = 0; slot_id < decision_scores.size(); slot_id++)
      {
        auto &slot = decision_scores[slot_id];
        auto action_id = slot[0].action;
        auto prob_chosen = slot[0].score;
        auto prob_to_click = user_slot_action_probabilities[chosen_user][actions_taken][slot_id];

        if (click_distribution(eng) < prob_to_click)
        {
          std::get<0>(clicks_impressions[chosen_user][actions_taken])[slot_id]++;
          outcomes.emplace_back(action_id, -1.f, prob_chosen);
        }
        else
        {
          outcomes.emplace_back(action_id, 0.f, prob_chosen);
        }
      }
      as_multiline(vw_ccb->l)->finish_example(*vw_ccb, ccb_ex_col);

      auto learn_ex = build_example_string_ccb(user_features[chosen_user], action_features, slot_features, outcomes);
      multi_ex learn_ex_col;
      for (auto str : learn_ex)
      {
        learn_ex_col.push_back(VW::read_example(*vw_ccb, str));
      }
      vw_ccb->learn(learn_ex_col);
      as_multiline(vw_ccb->l)->finish_example(*vw_ccb, learn_ex_col);
    }

    {
      std::vector<size_t> cb_actions_taken;
      std::vector<float> cb_probs;

      auto cb_ex_str = build_example_string_cb_no_slot(user_features[chosen_user], action_features);

      multi_ex cb_ex_col;
      for (auto str : cb_ex_str)
      {
        cb_ex_col.push_back(VW::read_example(*vw_cb, str));
      }

      vw_cb->predict(cb_ex_col);

      for (int slot_id = 0; slot_id < NUM_SLOTS; slot_id++)
      {
        auto action_score = cb_ex_col[0]->pred.a_s;
        cb_actions_taken.push_back(action_score[slot_id].action);
        cb_probs.push_back(action_score[slot_id].score);
      }

      as_multiline(vw_cb->l)->finish_example(*vw_cb, cb_ex_col);

      // Calculate reward for top action.
      std::vector<std::tuple<size_t, float, float>> cb_outcomes;
      for (auto slot_id = 0; slot_id < cb_actions_taken.size(); slot_id++)
      {
        auto prob_to_click = user_slot_action_probabilities[chosen_user][cb_actions_taken][slot_id];

        if (click_distribution(eng) < prob_to_click)
        {
          std::get<0>(cb_clicks_impressions[chosen_user][cb_actions_taken])[slot_id]++;
          cb_outcomes.emplace_back(cb_actions_taken[slot_id], -1.f, cb_probs[slot_id]);
        }
        else
        {
          cb_outcomes.emplace_back(cb_actions_taken[slot_id], 0.f, cb_probs[slot_id]);
        }
      }
      std::get<1>(cb_clicks_impressions[chosen_user][cb_actions_taken])++;

      // Learn from top action.
      auto learn_ex = build_example_string_cb_no_slot(user_features[chosen_user], action_features, cb_outcomes[0]);
      multi_ex cb_learn_ex_col;
      for (auto str : learn_ex)
      {
        cb_learn_ex_col.push_back(VW::read_example(*vw_cb, str));
      }
      vw_cb->learn(cb_learn_ex_col);
      as_multiline(vw_cb->l)->finish_example(*vw_cb, cb_learn_ex_col);
    }

    if (i % 5000 == 0)
    {
      // Clear terminal
      std::cout << "\033[2J" << std::endl;
      print_ctr(i, NUM_SLOTS, clicks_impressions);
      std::cout << "============================================== \n --CB-- " << std::endl;
      print_ctr(i, NUM_SLOTS, cb_clicks_impressions);
    }
  }

  std::cout << "\033[2J" << std::endl;
  print_ctr(NUM_ITER, NUM_SLOTS, clicks_impressions);
  std::cout << "============================================== \n --CB-- " << std::endl;
  print_ctr(NUM_ITER, NUM_SLOTS, cb_clicks_impressions);

  std::cout << "user,slot,action,count" << std::endl;
  for (auto user = 0; user < NUM_USERS; user++)
  {
    for (auto slot = 0; slot < NUM_SLOTS; slot++)
    {
      for (auto action = 0; action < NUM_ACTIONS; action++)
      {
        std::cout << user << "," << slot << "," << action << ","
                  << get_num_shown(user, slot, action, clicks_impressions) << std::endl;
      }
    }
  }

  std::cout << "\n\n\n--CB--\n"
            << std::endl;

  std::cout << "user,slot,action,count" << std::endl;
  for (auto user = 0; user < NUM_USERS; user++)
  {
    for (auto slot = 0; slot < NUM_SLOTS; slot++)
    {
      for (auto action = 0; action < NUM_ACTIONS; action++)
      {
        std::cout << user << "," << slot << "," << action << ","
                  << get_num_shown(user, slot, action, cb_clicks_impressions) << std::endl;
      }
    }
  }
  return 0;
}

int previous_action_dependent_ctr_2_slots_3_actions()
// int main()
{
  auto vw_ccb = VW::initialize("--ccb_explore_adf --epsilon 0.2 -l 0.001 --power_t 0 --quiet");
  auto vw_cb = VW::initialize("--cb_explore_adf --epsilon 0.2 -l 0.001 --power_t 0 --quiet --cb_sample --quadratic UA");

  auto const NUM_USERS = 3;
  auto const NUM_ACTIONS = 3;
  auto const NUM_SLOTS = 2;
  auto const NUM_ITER = 1000000;

  std::vector<std::string> user_features = {"a", "b", "c"};
  std::vector<std::string> action_features = {
      "d",
      "e",
      "f",
  };
  std::vector<std::string> slot_features = {
      "",
      "",
  };

  auto cb_clicks_impressions = generate_clicks_impressions_store(NUM_ACTIONS, NUM_SLOTS, NUM_USERS);
  auto clicks_impressions = generate_clicks_impressions_store(NUM_ACTIONS, NUM_SLOTS, NUM_USERS);

  // clang-format off
  std::vector<std::map<std::vector<size_t>,std::vector<float>>> user_slot_action_probabilities =
  {
    {
      {{0,1}, {0.01f, 0.8f}},
      {{1,0}, {0.01f, 0.01f}},
      {{0,2}, {0.01f, 0.01f}},
      {{2,0}, {0.01f, 0.01f}},
      {{1,2}, {0.01f, 0.01f}},
      {{2,1}, {0.01f, 0.01f}}
    },
    {
      {{0,1}, {0.01f, 0.01f}},
      {{1,0}, {0.01f, 0.01f}},
      {{0,2}, {0.01f, 0.01f}},
      {{2,0}, {0.01f, 0.01f}},
      {{1,2}, {0.01f, 0.8f}},
      {{2,1}, {0.01f, 0.01f}}
    },
    {
      {{0,1}, {0.01f, 0.01f}},
      {{1,0}, {0.01f, 0.01f}},
      {{0,2}, {0.01f, 0.8f}},
      {{2,0}, {0.01f, 0.01f}},
      {{1,2}, {0.01f, 0.01f}},
      {{2,1}, {0.01f, 0.01f}}
    }
};
  // clang-format on
  std::default_random_engine rd{0};
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> user_distribution(0, NUM_USERS - 1);
  std::uniform_real_distribution<float> click_distribution(0.0f, 1.0f);
  uint64_t merand_seed = 0;

  for (int i = 1; i <= NUM_ITER; i++)
  {
    auto chosen_user = user_distribution(eng);

    {
      // DO CCB
      auto ccb_ex_str = build_example_string_ccb(user_features[chosen_user], action_features, slot_features);

      multi_ex ccb_ex_col;
      for (auto str : ccb_ex_str)
      {
        ccb_ex_col.push_back(VW::read_example(*vw_ccb, str));
      }

      vw_ccb->predict(ccb_ex_col);

      std::vector<std::tuple<size_t, float, float>> outcomes;
      auto decision_scores = ccb_ex_col[0]->pred.decision_scores;

      std::vector<size_t> actions_taken;
      for (auto s : decision_scores)
      {
        actions_taken.push_back(s[0].action);
      };

      std::get<1>(clicks_impressions[chosen_user][actions_taken])++;
      for (auto slot_id = 0; slot_id < decision_scores.size(); slot_id++)
      {
        auto &slot = decision_scores[slot_id];
        auto action_id = slot[0].action;
        auto prob_chosen = slot[0].score;
        auto prob_to_click = user_slot_action_probabilities[chosen_user][actions_taken][slot_id];

        if (click_distribution(eng) < prob_to_click)
        {
          std::get<0>(clicks_impressions[chosen_user][actions_taken])[slot_id]++;
          outcomes.emplace_back(action_id, -1.f, prob_chosen);
        }
        else
        {
          outcomes.emplace_back(action_id, 0.f, prob_chosen);
        }
      }
      as_multiline(vw_ccb->l)->finish_example(*vw_ccb, ccb_ex_col);

      auto learn_ex = build_example_string_ccb(user_features[chosen_user], action_features, slot_features, outcomes);
      multi_ex learn_ex_col;
      for (auto str : learn_ex)
      {
        learn_ex_col.push_back(VW::read_example(*vw_ccb, str));
      }
      vw_ccb->learn(learn_ex_col);
      as_multiline(vw_ccb->l)->finish_example(*vw_ccb, learn_ex_col);
    }

    {
      std::vector<size_t> cb_actions_taken;
      std::vector<float> cb_probs;

      auto cb_ex_str = build_example_string_cb_no_slot(user_features[chosen_user], action_features);

      multi_ex cb_ex_col;
      for (auto str : cb_ex_str)
      {
        cb_ex_col.push_back(VW::read_example(*vw_cb, str));
      }

      vw_cb->predict(cb_ex_col);

      for (int slot_id = 0; slot_id < NUM_SLOTS; slot_id++)
      {
        auto action_score = cb_ex_col[0]->pred.a_s;
        cb_actions_taken.push_back(action_score[slot_id].action);
        cb_probs.push_back(action_score[slot_id].score);
      }

      as_multiline(vw_cb->l)->finish_example(*vw_cb, cb_ex_col);

      // Calculate reward for top action.
      std::vector<std::tuple<size_t, float, float>> cb_outcomes;
      for (auto slot_id = 0; slot_id < cb_actions_taken.size(); slot_id++)
      {
        auto prob_to_click = user_slot_action_probabilities[chosen_user][cb_actions_taken][slot_id];

        if (click_distribution(eng) < prob_to_click)
        {
          std::get<0>(cb_clicks_impressions[chosen_user][cb_actions_taken])[slot_id]++;
          cb_outcomes.emplace_back(cb_actions_taken[slot_id], -1.f, cb_probs[slot_id]);
        }
        else
        {
          cb_outcomes.emplace_back(cb_actions_taken[slot_id], 0.f, cb_probs[slot_id]);
        }
      }
      std::get<1>(cb_clicks_impressions[chosen_user][cb_actions_taken])++;

      // Learn from top action.
      auto learn_ex = build_example_string_cb_no_slot(user_features[chosen_user], action_features, cb_outcomes[0]);
      multi_ex cb_learn_ex_col;
      for (auto str : learn_ex)
      {
        cb_learn_ex_col.push_back(VW::read_example(*vw_cb, str));
      }
      vw_cb->learn(cb_learn_ex_col);
      as_multiline(vw_cb->l)->finish_example(*vw_cb, cb_learn_ex_col);
    }

    if (i % 5000 == 0)
    {
      // Clear terminal
      std::cout << "num_iter: " << i << "\n";
      std::cout << "ccb ctr: " << get_ctr(NUM_SLOTS, clicks_impressions) << "\n";
      std::cout << "cb ctr: " << get_ctr(NUM_SLOTS, cb_clicks_impressions) << "\n";
    }
  }

  std::cout << "num_iter: " << NUM_ITER << "\n";
  std::cout << "ccb ctr: " << get_ctr(NUM_SLOTS, clicks_impressions) << "\n";
  std::cout << "cb ctr: " << get_ctr(NUM_SLOTS, cb_clicks_impressions) << "\n";

  std::cout << "user,slot,action,count" << std::endl;
  for (auto user = 0; user < NUM_USERS; user++)
  {
    for (auto slot = 0; slot < NUM_SLOTS; slot++)
    {
      for (auto action = 0; action < NUM_ACTIONS; action++)
      {
        std::cout << user << "," << slot << "," << action << ","
                  << get_num_shown(user, slot, action, clicks_impressions) << std::endl;
      }
    }
  }

  std::cout << "\n\n\n--CB--\n"
            << std::endl;

  std::cout << "user,slot,action,count" << std::endl;
  for (auto user = 0; user < NUM_USERS; user++)
  {
    for (auto slot = 0; slot < NUM_SLOTS; slot++)
    {
      for (auto action = 0; action < NUM_ACTIONS; action++)
      {
        std::cout << user << "," << slot << "," << action << ","
                  << get_num_shown(user, slot, action, cb_clicks_impressions) << std::endl;
      }
    }
  }
  return 0;
}
