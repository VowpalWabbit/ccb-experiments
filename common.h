#pragma once

#include <iostream>

#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <map>
#include <numeric>
#include <exception>
#include <algorithm>

std::vector<std::string> build_example_string_ccb(std::string& user_feature, std::vector<std::string>& action_features,
    std::vector<std::string>& slot_features, std::vector<std::tuple<size_t, float, float>> labels = {})
{
  std::vector<std::string> ret_val;
  std::stringstream ss;
  ss << "ccb shared |User " << user_feature;
  ret_val.push_back(ss.str());
  ss.str(std::string());

  for (auto action : action_features)
  {
    ss << "ccb action |Action " << action;
    ret_val.push_back(ss.str());
    ss.str(std::string());
  }

  for (size_t i = 0; i < slot_features.size(); i++)
  {
    ss << "ccb slot ";
    if (labels.size() > i)
    {
      ss << (std::get<0>(labels[i])) << ":" << std::get<1>(labels[i]) << ":" << std::get<2>(labels[i]);
    }
    ss << " |Slot " << slot_features[i];
    ret_val.push_back(ss.str());
    ss.str(std::string());
  }

  return ret_val;
}

std::vector<std::string> build_example_string_ccb(std::string& user_feature, std::vector<std::string>& action_features,
  size_t num_slots, std::vector<std::tuple<size_t, float, float>> labels = {})
{
  std::vector<std::string> ret_val;
  std::stringstream ss;
  ss << "ccb shared |User " << user_feature;
  ret_val.push_back(ss.str());
  ss.str(std::string());

  for (auto action : action_features)
  {
    ss << "ccb action |Action " << action;
    ret_val.push_back(ss.str());
    ss.str(std::string());
  }

  for (size_t i = 0; i < num_slots; i++)
  {
    ss << "ccb slot ";
    if (labels.size() > i)
    {
      ss << (std::get<0>(labels[i])) << ":" << std::get<1>(labels[i]) << ":" << std::get<2>(labels[i]);
    }
    ss << " |";
    ret_val.push_back(ss.str());
    ss.str(std::string());
  }

  return ret_val;
}

std::vector<std::string> build_example_string_cb(std::string& user_feature, std::vector<std::string>& action_features,
    std::string& slot_features, std::tuple<size_t, float, float> label = std::tuple<size_t, float, float>{1, -1.f, -1.f})
{
  std::vector<std::string> ret_val;
  std::stringstream ss;
  ss << "shared |User " << user_feature << " |Slot " << slot_features;
  ret_val.push_back(ss.str());
  ss.str(std::string());

  for (size_t i = 0; i < action_features.size(); i++)
  {
    if (std::get<0>(label) == i)
    {
      ss << (std::get<0>(label) + 1) << ":" << std::get<1>(label) << ":" << std::get<2>(label);
    }
    ss << " |Action  " << action_features[i];
    ret_val.push_back(ss.str());
    ss.str(std::string());
  }

  return ret_val;
}

std::vector<std::string> build_example_string_cb_no_slot(std::string& user_feature,
    std::vector<std::string>& action_features, std::tuple<size_t, float, float> label = std::tuple<size_t, float, float>{-1, -1.f, -1.f})
{
  std::vector<std::string> ret_val;
  std::stringstream ss;
  ss << "shared |User " << user_feature;
  ret_val.push_back(ss.str());
  ss.str(std::string());

  for (size_t i = 0; i < action_features.size(); i++)
  {
    if (std::get<0>(label) == i)
    {
      ss << (std::get<0>(label) + 1) << ":" << std::get<1>(label) << ":" << std::get<2>(label);
    }
    ss << " |Action " << action_features[i];
    ret_val.push_back(ss.str());
    ss.str(std::string());
  }

  return ret_val;
}

void print_click_shows(size_t num_iter,
    std::vector<std::map<std::vector<size_t>, std::tuple<std::vector<size_t>, size_t>>>& clicks_impressions)
{
  std::cout << "num iterations: " << num_iter << "\n";
  std::cout << "user\tactions\tclicks          shown\tctr\n";
  size_t total_clicks = 0;
  size_t total_shown = 0;
  for (auto user_index = 0; user_index < clicks_impressions.size(); user_index++)
  {
    std::cout << "--\n";

    for (auto& kv : clicks_impressions[user_index])
    {
      std::cout << user_index << "\t";

      // actions
      for (auto num : kv.first)
      {
        std::cout << num;
      }
      std::cout << "\t";

      // clicks
      std::stringstream ss;

      for (auto num : std::get<0>(kv.second))
      {
        ss << num << ",";
        total_clicks += num;
      }

      std::cout << std::setw(16) << std::left << ss.str();

      // shown
      std::cout << std::get<1>(kv.second) << "\t";
      total_shown += std::get<1>(kv.second);

      for (auto& num : std::get<0>(kv.second))
      {
        std::cout << std::setprecision(4) << (float)num / std::get<1>(kv.second) << ",";
      }

      std::cout << "\n";
    }
  }
  // Total CTR
  std::cout << "clicks: " << total_clicks << "\n";
  std::cout << "shows: " << total_shown << "\n";
  std::cout << std::endl;
}

size_t get_num_shown(size_t user, size_t slot, size_t action,
    std::vector<std::map<std::vector<size_t>, std::tuple<std::vector<size_t>, size_t>>>& clicks_impressions)
{
  size_t total = 0;
  for (auto& kv : clicks_impressions[user])
  {
    if (kv.first[slot] == action)
    {
      total += std::get<1>(kv.second);
    }
  }
  return total;
}


float get_ctr(size_t num_slots,
    std::vector<std::map<std::vector<size_t>, std::tuple<std::vector<size_t>, size_t>>>& clicks_impressions)
{
  size_t total_clicks = 0;
  size_t total_shown = 0;
  for (auto user_index = 0; user_index < clicks_impressions.size(); user_index++)
  {
    for (auto& kv : clicks_impressions[user_index])
    {
      for (auto num : std::get<0>(kv.second))
      {
        total_clicks += num;
      }

      // shown
      total_shown += std::get<1>(kv.second);
    }
  }

  return ((float)total_clicks / (float)total_shown) / (float)num_slots;
}

void print_ctr(size_t num_iter, size_t num_slots,
    std::vector<std::map<std::vector<size_t>, std::tuple<std::vector<size_t>, size_t>>>& clicks_impressions)
{
  std::cout << "num iterations: " << num_iter << "\n";
  size_t total_clicks = 0;
  size_t total_shown = 0;
  for (auto user_index = 0; user_index < clicks_impressions.size(); user_index++)
  {
    for (auto& kv : clicks_impressions[user_index])
    {
      for (auto num : std::get<0>(kv.second))
      {
        total_clicks += num;
      }

      // shown
      total_shown += std::get<1>(kv.second);
    }
  }
  // Total CTR
  std::cout << "clicks: " << total_clicks << "\n";
  std::cout << "shows: " << total_shown << "\n";
  std::cout << "ctr: " << ((float)total_clicks / (float)total_shown) / (float)num_slots << "\n";
  std::cout << std::endl;
}

void print_ctr_as_csv(size_t num_iter, size_t num_slots,
    std::vector<std::map<std::vector<size_t>, std::tuple<std::vector<size_t>, size_t>>>& ccb_clicks_impressions,
    std::vector<std::map<std::vector<size_t>, std::tuple<std::vector<size_t>, size_t>>>& cb_clicks_impressions)
{
  std::cout << num_iter << "," << get_ctr(num_slots, ccb_clicks_impressions) << ","
            << get_ctr(num_slots, cb_clicks_impressions) << std::endl;
}

void print_click_shows_as_csv(size_t num_iter,
    std::vector<std::map<std::vector<size_t>, std::tuple<std::vector<size_t>, size_t>>>& clicks_impressions)
{
  std::cout << "num iterations: " << num_iter << "\n";
  std::cout << "user,actions,clicks,shown,ctr\n";

  for (auto user_index = 0; user_index < clicks_impressions.size(); user_index++)
  {
    std::cout << "--\n";

    for (auto& kv : clicks_impressions[user_index])
    {
      std::cout << user_index << ",";

      // actions
      for (auto num : kv.first)
      {
        std::cout << num;
      }
      std::cout << ",";

      // clicks
      std::stringstream ss;
      for (auto num : std::get<0>(kv.second))
      {
        ss << num << ",";
      }

      std::cout << "\"" << ss.str() << "\",";

      // shown
      std::cout << std::get<1>(kv.second) << ",\"";

      for (auto num : std::get<0>(kv.second))
      {
        std::cout << std::setprecision(4) << (float)num / std::get<1>(kv.second) << ",";
      }
      std::cout << "\"\n";
    }
  }
  std::cout << std::endl;
}

std::vector<std::vector<size_t>> permutations_pick_k(std::vector<size_t> items, size_t num_to_pick)
{
  if (items.size() == 0 || num_to_pick == 0)
    return {{}};

  std::vector<std::vector<size_t>> result;

  for (auto item : items)
  {
    auto remainder = items;
    remainder.erase(std::find(remainder.begin(), remainder.end(), item));

    auto partials = permutations_pick_k(remainder, num_to_pick - 1);
    for (auto partial : partials)
    {
      partial.push_back(item);
      result.push_back(partial);
    }
  }

  return result;
}

std::vector<std::map<std::vector<size_t>, std::tuple<std::vector<size_t>, size_t>>> generate_clicks_impressions_store(
    size_t num_actions, size_t num_slots, size_t num_users)
{
  if (num_slots > num_actions)
    throw std::runtime_error("num_slots is greater than num_actions");

  std::vector<std::map<std::vector<size_t>, std::tuple<std::vector<size_t>, size_t>>> result;

  std::vector<size_t> actions(num_actions);
  std::iota(std::begin(actions), std::end(actions), 0);

  auto slot_permutations = permutations_pick_k(actions, num_slots);

  for (auto i = 0; i < num_users; i++)
  {
    std::map<std::vector<size_t>, std::tuple<std::vector<size_t>, size_t>> current;
    for (auto& slot_permutation : slot_permutations)
    {
      current.insert({slot_permutation, std::tuple<std::vector<size_t>, size_t>{std::vector<size_t>(num_slots, 0), 0}});
    }
    result.push_back(current);
  }

  return result;
}

std::map<std::vector<size_t>, std::vector<float>> generate_slot_dependent_probs(
    size_t num_actions, size_t num_slots, std::vector<std::vector<float>> slot_action_probs)
{
  std::vector<size_t> actions(num_actions);
  std::iota(std::begin(actions), std::end(actions), 0);

  auto slot_permutations = permutations_pick_k(actions, num_slots);
  std::map<std::vector<size_t>, std::vector<float>> result;
  for (auto& slot_permutation : slot_permutations)
  {
    std::vector<float> current;
    for (auto slot_index = 0; slot_index < slot_permutation.size(); slot_index++)
    {
      current.push_back(slot_action_probs[slot_index][slot_permutation[slot_index]]);
    }
    result.insert({slot_permutation, current});
  }
  return result;
}
