from setuptools import setup, find_packages

setup(
        name="tweets-sentiment",
        version="1.0",
        author="Novica Sarenac, Predrag Njegovanovic, Sasa Ferenc",
        author_email="novicasarenac@gmail.com, djaps94@gmail.com",
        description="Sentiment analysis of Twitter posts and comments",
        license="MIT",
        url="https://github.com/novicasarenac/tweets-sentiment",
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            "numpy",
            "scipy",
            "matplotlib"
        ]
    )
