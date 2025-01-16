<h1 id="projects"></h1>

<h2 style="margin: 60px 0px -15px;">Projects <temp style="font-size:15px;"></temp></h2>


<div class="projects">
<ol class="bibliography">


{% for link in site.data.projects.main %}

<li>
<div class="pub-row">
  <div class="col-sm-9" style="position: relative;padding-right: 15px;padding-left: 20px;padding-top: 20px;">
      <h3><a href="{{ link.pdf }}">{{ link.title }}</a></h3>
      <div class="col-sm-3 abbr" style="position: relative;padding-right: 15px;padding-left: 15px;">
        <img src="{{ link.image }}" class="teaser img-fluid z-depth-1" style="width:{{ link.width }}; height:auto;display: block; margin-left: auto; margin-right: auto;">
      </div>
    <div class="description" style="text-align: justify; text-indent: 0px;"><strong>Description:</strong> {{ link.description }}</div>
    <div class="keywords"><strong>Keywords:</strong> {{ link.keywords }}</div>

    {% if link.video %}
      <p>Video URL: {{ link.video }}</p>
      <div class="video">
        <iframe width="560" height="315" src="{{ link.video }}" title="YouTube video player" frameborder="1" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
      </div>
    {% endif %}

    <div class="links">
      {% if link.pdf %} 
      <a href="{{ link.pdf }}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">PDF</a>
      {% endif %}
      {% if link.code %} 
      <a href="{{ link.code }}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">Code</a>
      {% endif %}
      {% if link.page %} 
      <a href="{{ link.page }}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">Project Page</a>
      {% endif %}
      {% if link.bibtex %} 
      <a href="{{ link.bibtex }}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">BibTex</a>
      {% endif %}
      {% if link.notes %} 
      <strong> <i style="color:#e74d3c">{{ link.notes }}</i></strong>
      {% endif %}
      {% if link.others %} 
      {{ link.others }}
      {% endif %}
    </div>
  </div>
</div>
</li>

<br>

{% endfor %}

</ol>
</div>


