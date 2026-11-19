# Piauí Resiliente: site publicado

Esta branch guarda só os arquivos prontos do site (HTML, CSS e JavaScript estáticos). A Vercel publica a pasta como está, sem build.

Não edite estes arquivos à mão. Eles são gerados na branch de trabalho do projeto:

    python -m src.data.build_demo      # só se os dados mudaram
    python -m src.web.build_site --saida site

Para atualizar o site publicado, copie o conteúdo de `site/` para a raiz desta branch, faça o commit e envie (`git push`). A Vercel publica sozinha.

O site continua em MVP experimental, em calibração com S2iD/TCE-PI.
