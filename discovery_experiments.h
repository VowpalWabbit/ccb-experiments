#pragma once

int discovery_rate()
{
  auto const NUM_USERS = 1;
  auto const NUM_ACTIONS = 10;
  auto const NUM_SLOTS = 10;
  auto const NUM_ITER = 1000;
  auto const NUM_SEEDS = 20;

  std::vector<std::vector<size_t>> top_ccb = std::vector<std::vector<size_t>>(20, std::vector<size_t>(NUM_ITER + 1, 0));
  std::vector<std::vector<size_t>> top_cb = std::vector<std::vector<size_t>>(20, std::vector<size_t>(NUM_ITER + 1, 0));

  uint32_t seed = 1;
  for (int seed_i = 0; seed_i < NUM_SEEDS; seed_i++)
  {
    seed += 9542234;

    auto vw_ccb = VW::initialize(
        "--ccb_explore_adf --epsilon 0.2 -l 0.001 --power_t 0 --quiet -q UA --random_seed " + std::to_string(seed));
    auto vw_cb = VW::initialize(
        "--cb_explore_adf --epsilon 0.2 -l 0.001 --power_t 0 --quiet --cb_sample --quadratic UA --random_seed " +
        std::to_string(seed));

    std::vector<std::string> user_features = {"user1"};
    std::vector<std::string> action_features = {"a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"};

    std::vector<float> action_probs = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.8f, 0.1f, 0.1f, 0.1f};

    auto const TOP_INDEX = 6;
    auto top_action_shows_ccb = 0;
    auto top_action_shows_cb = 0;

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

        if (actions_taken[0] == TOP_INDEX)
          top_action_shows_ccb++;

        top_ccb[seed_i][i] = top_action_shows_ccb;

        for (auto slot_id = 0; slot_id < decision_scores.size(); slot_id++)
        {
          auto &slot = decision_scores[slot_id];
          auto action_id = slot[0].action;
          auto prob_chosen = slot[0].score;
          auto prob_to_click = action_probs[actions_taken[slot_id]];

          if (click_distribution(eng) < prob_to_click)
          {
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

      // DO CB Ranking
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

        if (cb_actions_taken[0] == TOP_INDEX)
          top_action_shows_cb++;

        top_cb[seed_i][i] = top_action_shows_cb;

        as_multiline(vw_cb->l)->finish_example(*vw_cb, cb_ex_col);

        // Calculate reward for top action.
        std::vector<std::tuple<size_t, float, float>> cb_outcomes;
        for (auto slot_id = 0; slot_id < cb_actions_taken.size(); slot_id++)
        {
          auto prob_to_click = action_probs[cb_actions_taken[slot_id]];

          if (click_distribution(eng) < prob_to_click)
          {
            cb_outcomes.emplace_back(cb_actions_taken[slot_id], -1.f, cb_probs[slot_id]);
          }
          else
          {
            cb_outcomes.emplace_back(cb_actions_taken[slot_id], 0.f, cb_probs[slot_id]);
          }
        }

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
    }
  }

  for (int i = 0; i < NUM_ITER; i++)
  {
    std::cout << i << ",";
    for (int j = 0; j < NUM_SEEDS; j++)
    {
      std::cout << top_ccb[j][i] << ",";
    }
    std::cout << "\n";
  }

  std::cout << "CB ----\n";

  for (int i = 1; i <= NUM_ITER; i++)
  {
    std::cout << i << ",";
    for (int j = 0; j < NUM_SEEDS; j++)
    {
      std::cout << top_cb[j][i] << ",";
    }
    std::cout << "\n";
  }
  return 0;
}

int progressive_ctr(bool coin)
{
  auto const NUM_USERS = 1;
  auto const NUM_ACTIONS = 10;
  auto const NUM_SLOTS = 5;
  auto const NUM_ITER = 10000;
  auto const NUM_SEEDS = 20;

  std::vector<std::vector<std::pair<int,int>>> str_over_time_per_slot_ccb(NUM_SEEDS, std::vector<std::pair<int,int>>());
  std::vector<std::vector<std::pair<int,int>>> str_over_time_per_slot_cb(NUM_SEEDS, std::vector<std::pair<int,int>>());

  uint32_t seed = 1;
  for (int seed_i = 0; seed_i < NUM_SEEDS; seed_i++)
  {
    seed += 9542234;

    const std::string args = coin ? "--coin": "-l 0.001 --power_t 0";

    auto vw_ccb = VW::initialize(
        "--ccb_explore_adf --epsilon 0.04 " + args + "  --quiet -q UA --random_seed " + std::to_string(seed));
    auto vw_cb = VW::initialize(
        "--cb_explore_adf --epsilon 0.2 " + args +" --quiet  --cb_sample  --quadratic UA --random_seed " +
        std::to_string(seed));

    std::vector<std::string> user_features = {"user1"};
    std::vector<std::string> action_features = {"a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"};

    std::vector<float> action_probs = {0.01f, 0.1f, 0.01f, 0.1f, 0.01f, 0.01f, 0.3f, 0.01f, 0.01f, 0.1f};

    std::default_random_engine rd{0};
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> user_distribution(0, NUM_USERS - 1);
    std::uniform_real_distribution<float> click_distribution(0.0f, 1.0f);
    uint64_t merand_seed = 0;

    int shows_ccb = 0;
    int clicks_ccb = 0;

    int shows_cb = 0;
    int clicks_cb = 0;
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

        for (auto slot_id = 0; slot_id < decision_scores.size(); slot_id++)
        {
          auto &slot = decision_scores[slot_id];
          auto action_id = slot[0].action;
          auto prob_chosen = slot[0].score;
          auto prob_to_click = action_probs[actions_taken[slot_id]];

          shows_ccb++;
          if (click_distribution(eng) < prob_to_click)
          {
            clicks_ccb++;
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

        str_over_time_per_slot_ccb[seed_i].push_back({shows_ccb, clicks_ccb});
      }

      // DO CB Ranking
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
          auto prob_to_click = action_probs[cb_actions_taken[slot_id]];

          shows_cb++;
          if (click_distribution(eng) < prob_to_click)
          {
            clicks_cb++;
            cb_outcomes.emplace_back(cb_actions_taken[slot_id], -1.f, cb_probs[slot_id]);
          }
          else
          {
            cb_outcomes.emplace_back(cb_actions_taken[slot_id], 0.f, cb_probs[slot_id]);
          }
        }

        // Learn from top action.
        auto learn_ex = build_example_string_cb_no_slot(user_features[chosen_user], action_features, cb_outcomes[0]);
        multi_ex cb_learn_ex_col;
        for (auto str : learn_ex)
        {
          cb_learn_ex_col.push_back(VW::read_example(*vw_cb, str));
        }
        vw_cb->learn(cb_learn_ex_col);
        as_multiline(vw_cb->l)->finish_example(*vw_cb, cb_learn_ex_col);

        str_over_time_per_slot_cb[seed_i].push_back({shows_cb, clicks_cb});
      }
    }
  }

  // Print results in csv
  std::cout << "i,";
  for (int i = 0; i < NUM_SEEDS; i++)
  {
    std::cout << "ccb" << i << ",";
  }

  for (int i = 0; i < NUM_SEEDS; i++)
  {
    std::cout << "cb" << i << ",";
  }
  std::cout << std::endl;

  for(int i = 0 ; i < NUM_ITER; i += 10)
  {
    std::cout << i << ",";
    for (int seed_i = 0; seed_i < NUM_SEEDS; seed_i++)
    {
      std::cout << (float)str_over_time_per_slot_ccb[seed_i][i].second / (float)str_over_time_per_slot_ccb[seed_i][i].first << ",";
    }
    for (int seed_i = 0; seed_i < NUM_SEEDS; seed_i++)
    {
      std::cout << (float)str_over_time_per_slot_cb[seed_i][i].second / (float)str_over_time_per_slot_ccb[seed_i][i].first << ",";
    }
    std::cout << std::endl;
  }
  return 0;
}


int smaller_subset_discovery()
{
  auto const NUM_USERS = 1;
  auto const NUM_ACTIONS = 10;
  auto const NUM_SLOTS = 3;
  auto const NUM_ITER = 10000;
  auto const NUM_SEEDS = 20;

  std::vector<std::vector<size_t>> top_ccb = std::vector<std::vector<size_t>>(20, std::vector<size_t>(NUM_ITER + 1, 0));
  std::vector<std::vector<size_t>> top_cb = std::vector<std::vector<size_t>>(20, std::vector<size_t>(NUM_ITER + 1, 0));

  uint32_t seed = 1;
  for (int seed_i = 0; seed_i < NUM_SEEDS; seed_i++)
  {
    seed += 9542234;

    auto cb_clicks_impressions = generate_clicks_impressions_store(NUM_ACTIONS, NUM_SLOTS, NUM_USERS);
    auto clicks_impressions = generate_clicks_impressions_store(NUM_ACTIONS, NUM_SLOTS, NUM_USERS);

    auto vw_ccb = VW::initialize(
        "--ccb_explore_adf --epsilon 0.06 -l 0.001 --power_t 0 --quiet -q UA --random_seed " + std::to_string(seed));
    auto vw_cb = VW::initialize(
        "--cb_explore_adf --epsilon 0.2 -l 0.001 --power_t 0 --quiet --cb_sample --quadratic UA --random_seed " +
        std::to_string(seed));

    std::vector<std::string> user_features = {"user1"};
    std::vector<std::string> action_features = {"a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"};

    std::vector<float> action_probs = {0.03f, 0.03f, 0.03f, 0.07f, 0.03f, 0.08f, 0.03f, 0.03f, 0.1f, 0.03f};

    auto const TOP_INDEX = 5;
    auto top_action_shows_ccb = 0;
    auto top_action_shows_cb = 0;

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

        if (actions_taken[0] == TOP_INDEX)
          top_action_shows_ccb++;

        top_ccb[seed_i][i] = top_action_shows_ccb;

        for (auto slot_id = 0; slot_id < decision_scores.size(); slot_id++)
        {
          auto &slot = decision_scores[slot_id];
          auto action_id = slot[0].action;
          auto prob_chosen = slot[0].score;
          auto prob_to_click = action_probs[actions_taken[slot_id]];

          std::get<1>(clicks_impressions[chosen_user][actions_taken])++;
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

      // DO CB Ranking
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

        if (cb_actions_taken[0] == TOP_INDEX)
          top_action_shows_cb++;

        top_cb[seed_i][i] = top_action_shows_cb;

        as_multiline(vw_cb->l)->finish_example(*vw_cb, cb_ex_col);

        // Calculate reward for top action.
        std::vector<std::tuple<size_t, float, float>> cb_outcomes;
        for (auto slot_id = 0; slot_id < cb_actions_taken.size(); slot_id++)
        {
          auto prob_to_click = action_probs[cb_actions_taken[slot_id]];

          std::get<1>(cb_clicks_impressions[chosen_user][cb_actions_taken])++;
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
    }

    std::cout << seed << ",";
    print_ctr_as_csv(NUM_ITER, NUM_SLOTS, clicks_impressions, cb_clicks_impressions);
  }

  for (int i = 0; i < NUM_ITER; i++)
  {
    std::cout << i << ",";
    for (int j = 0; j < NUM_SEEDS; j++)
    {
      std::cout << top_ccb[j][i] << ",";
    }
    std::cout << "\n";
  }

  std::cout << "CB ----\n";

  for (int i = 1; i <= NUM_ITER; i++)
  {
    std::cout << i << ",";
    for (int j = 0; j < NUM_SEEDS; j++)
    {
      std::cout << top_cb[j][i] << ",";
    }
    std::cout << "\n";
  }
  return 0;
}
