#pragma once

int diversity()
{
  auto vw_ccb = VW::initialize("--ccb_explore_adf --epsilon 0.2 --quadratic UA -l 0.001 --power_t 0 --quiet");
  auto vw_cb = VW::initialize("--cb_explore_adf --epsilon 0.2 -l 0.001 --power_t 0 --quiet --cb_sample --quadratic UA");

  auto const NUM_USERS = 1;
  auto const NUM_ACTIONS = 3;
  auto const NUM_SLOTS = 3;
  auto const NUM_ITER = 1000000;

  std::vector<std::string> user_features = {"a"};
  std::vector<std::string> action_features = {"car1 car", "car2 car", "cat1 cat"};

  auto cb_clicks_impressions = generate_clicks_impressions_store(NUM_ACTIONS, NUM_SLOTS, NUM_USERS);
  auto clicks_impressions = generate_clicks_impressions_store(NUM_ACTIONS, NUM_SLOTS, NUM_USERS);

  std::vector<std::map<std::vector<size_t>, std::vector<float>>> user_slot_action_probabilities = {
      {{{0, 1, 2}, {0.5f, 0.1f, 0.3f}}, {{1, 0, 2}, {0.5f, 0.1f, 0.3f}}, {{2, 0, 1}, {0.3f, 0.5f, 0.1f}},
          {{0, 2, 1}, {0.5f, 0.3f, 0.1f}}, {{1, 2, 0}, {0.5f, 0.3f, 0.1f}}, {{2, 1, 0}, {0.3f, 0.5f, 0.1f}}}};

  std::default_random_engine rd{0};
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> user_distribution(0, NUM_USERS - 1);
  std::uniform_real_distribution<float> click_distribution(0.0f, 1.0f);

  for (int i = 1; i <= NUM_ITER; i++)
  {
    // auto chosen_user = user_distribution(eng);
    auto chosen_user = 0;

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
        auto& slot = decision_scores[slot_id];
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
      std::cout << "\033[2J" << std::endl;
      print_click_shows_as_csv(i, clicks_impressions);
      std::cout << "============================================== \n --CB-- " << std::endl;
      print_click_shows_as_csv(i, cb_clicks_impressions);
    }
  }

  std::cout << "\033[2J" << std::endl;
  print_click_shows_as_csv(NUM_ITER, clicks_impressions);
  std::cout << "============================================== \n --CB-- " << std::endl;
  print_click_shows_as_csv(NUM_ITER, cb_clicks_impressions);
  return 0;
}


int diversity_with_interest_vectors()
{
  auto vw_ccb = VW::initialize("--ccb_explore_adf --epsilon 0.2 -l 0.001 --power_t 0 --quiet");
  auto vw_cb = VW::initialize("--cb_explore_adf --epsilon 0.2 -l 0.001 --power_t 0 --quiet --cb_sample --quadratic UA");

  auto const NUM_USERS = 3;
  auto const NUM_ACTIONS = 7;
  auto const NUM_SLOTS = 3;
  auto const NUM_ITER = 1000000;

  std::vector<std::string> user_features = {"a", "b", "c"};
  std::vector<std::string> action_features = {"a1", "a2", "a3", "a4", "a5", "a6", "a7"};
  std::vector<std::string> slot_features = {"h", "i", "j"};

  // Topic 1, topic 2, topic 3, topic 4
  std::vector<std::vector<float>> user_interest = {
    {
      0.4f,0.0f,0.3f,0.1f
    },
    {
      0.1f,0.7f,0.0f,0.1f
    }
    ,
    {
      0.1f,0.1f,0.1f,0.6f
    }
  };


  std::vector<std::vector<float>> action_interest = {
    {
      0.4f,0.0f,0.0f,0.0f
    },
    {
      0.6f,0.1f,0.0f,0.0f
    },
    {
      0.0f,0.7f,0.0f,0.0f
    },
    {
      0.0f,0.9f,0.0f,0.1f
    }
    ,
    {
      0.0f,0.0f,0.9f,0.0f
    }
    ,
    {
      0.0f,0.0f,0.7f,0.0f
    },
    {
      0.0f,0.0f,0.0f,0.6f
    }
  };

  auto cb_clicks_impressions = generate_clicks_impressions_store(NUM_ACTIONS, NUM_SLOTS, NUM_USERS);
  auto clicks_impressions = generate_clicks_impressions_store(NUM_ACTIONS, NUM_SLOTS, NUM_USERS);

  std::default_random_engine rd{0};
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> user_distribution(0, NUM_USERS - 1);

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
        auto& slot = decision_scores[slot_id];
        auto action_id = slot[0].action;
        auto prob_chosen = slot[0].score;
        auto cross = std::inner_product(user_interest[chosen_user].begin(),user_interest[chosen_user].end(), action_interest[action_id].begin(), 0.f);

        if (cross > 0.1)
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
        auto cross = std::inner_product(user_interest[chosen_user].begin(),user_interest[chosen_user].end(), action_interest[cb_actions_taken[slot_id]].begin(), 0.f);


        if (cross > 0.1)
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
      print_click_shows_as_csv(i, clicks_impressions);
      std::cout << "============================================== \n --CB-- " << std::endl;
      print_click_shows_as_csv(i, cb_clicks_impressions);

      print_ctr(i, NUM_SLOTS, clicks_impressions);
      print_ctr(i, NUM_SLOTS, cb_clicks_impressions);
    }
  }

  std::cout << "\033[2J" << std::endl;
  print_click_shows_as_csv(NUM_ITER, clicks_impressions);
  std::cout << "============================================== \n --CB-- " << std::endl;
  print_click_shows_as_csv(NUM_ITER, cb_clicks_impressions);
  return 0;
}
