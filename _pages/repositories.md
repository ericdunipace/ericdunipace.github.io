---
layout: page
permalink: /repositories/
title: repositories
description:
nav: true
nav_order: 4
---

{% if site.data.repositories.github_repos %}

<div class="repo-grid">
  {% for repo in site.data.repositories.github_repos %}
    {% include repository/repo.liquid repository=repo %}
  {% endfor %}
</div>
{% endif %}

{% for user in site.data.repositories.github_users %}

<p class="repo-more"><a href="https://github.com/{{ user }}">All repositories on GitHub →</a></p>
{% endfor %}
